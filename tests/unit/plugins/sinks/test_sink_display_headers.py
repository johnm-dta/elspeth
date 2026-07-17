"""Tests for sink header output functionality.

Tests the unified headers configuration option for CSV and JSON sinks
that controls output header naming (normalized, original, or custom mapping).
"""

import json
from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


class _FieldResolutionLandscape:
    def __init__(self, *outcomes: dict[str, str] | None | Exception) -> None:
        self._outcomes = list(outcomes)
        self.get_source_field_resolution_call_count = 0

    def get_source_field_resolution(self, _run_id: str) -> dict[str, str] | None:
        self.get_source_field_resolution_call_count += 1
        if not self._outcomes:
            return None
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestCSVSinkHeaders:
    """Tests for CSVSink header output functionality."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())


class TestJSONSinkHeaders:
    """Tests for JSONSink header output functionality."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())


class TestCSVCustomHeadersAppendMode:
    """Tests for CSV append mode with custom headers."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())


class TestCSVCustomHeadersSpecialCharacters:
    """Tests for CSV custom headers containing special characters."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())


class TestJSONLCustomHeadersAppendMode:
    """Tests for JSONL append/resume mode with custom headers."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_resume_validation_with_custom_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Resume validation succeeds when existing file uses custom header names."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # Pre-create file with custom headers
        with open(output_file, "w") as f:
            f.write(json.dumps({"User ID": "u1", "Amount": 100.0}) + "\n")

        # Open in append mode with matching custom headers - should validate successfully
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Amount"},
                    "mode": "append",
                }
            )
        )

        # Validation happens lazily, trigger it by calling validate_output_target
        result = sink.validate_output_target()
        assert result.valid, f"Validation failed: {result.error_message}"


class TestResumeValidationWithOriginalHeaders:
    """Tests for resume validation when headers: original is enabled.

    This tests the scenario where:
    1. A run completes with headers: original (output has source header names)
    2. User runs `elspeth resume` on the same run
    3. validate_output_target() must correctly compare existing display names
       against expected display names (not normalized schema names)

    The fix requires providing the field resolution mapping to sinks BEFORE
    calling validate_output_target() during resume.
    """

    def test_csv_resume_validation_with_original_headers(self, tmp_path: Path) -> None:
        """CSV resume validation succeeds when original headers mapping is provided."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"

        # Pre-create CSV file with original headers (as if previous run used headers: original)
        with open(output_file, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["User ID", "Amount (USD)"])  # Original names
            writer.writerow(["u1", "100.0"])

        # Create sink with headers: original (resume scenario)
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                    "headers": "original",
                }
            )
        )

        # Simulate resume: provide the field resolution mapping BEFORE validation
        # This is what the CLI will do during `elspeth resume`
        field_resolution = {
            "User ID": "user_id",
            "Amount (USD)": "amount_usd",
        }
        sink.set_resume_field_resolution(field_resolution)

        # Now validation should succeed - it can map schema fields to display names
        result = sink.validate_output_target()
        assert result.valid, f"Validation failed: {result.error_message}"

    def test_csv_resume_validation_without_resolution_fails(self, tmp_path: Path) -> None:
        """CSV resume validation fails closed when headers: original lacks mapping."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"

        # Pre-create CSV file with display headers
        with open(output_file, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["User ID", "Amount (USD)"])
            writer.writerow(["u1", "100.0"])

        # Create sink with headers: original but don't provide resolution
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                    "headers": "original",
                }
            )
        )

        result = sink.validate_output_target()
        assert not result.valid, "Should fail when headers: original but no resolution"
        assert result.error_message is not None
        assert "field resolution" in result.error_message

    def test_jsonl_resume_validation_with_original_headers(self, tmp_path: Path) -> None:
        """JSONL resume validation succeeds when original headers mapping is provided."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # Pre-create JSONL file with original headers
        with open(output_file, "w") as f:
            f.write(json.dumps({"User ID": "u1", "Amount (USD)": 100.0}) + "\n")

        # Create sink with headers: original
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                    "headers": "original",
                }
            )
        )

        # Provide field resolution for resume
        field_resolution = {
            "User ID": "user_id",
            "Amount (USD)": "amount_usd",
        }
        sink.set_resume_field_resolution(field_resolution)

        # Now validation should succeed
        result = sink.validate_output_target()
        assert result.valid, f"Validation failed: {result.error_message}"

    def test_jsonl_resume_validation_without_resolution_fails(self, tmp_path: Path) -> None:
        """JSONL resume validation fails when headers: original but no resolution."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # Pre-create JSONL file with original headers
        with open(output_file, "w") as f:
            f.write(json.dumps({"User ID": "u1", "Amount (USD)": 100.0}) + "\n")

        # Create sink with headers: original but no resolution
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount_usd: float"]},
                    "headers": "original",
                }
            )
        )

        # Without resolution, validation should fail
        result = sink.validate_output_target()
        assert not result.valid, "Should fail when headers: original but no resolution"

    def test_csv_resume_validation_strict_mode_with_original_headers(self, tmp_path: Path) -> None:
        """Strict mode resume validation works with headers: original."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"

        with open(output_file, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["User ID", "Status"])
            writer.writerow(["u1", "active"])

        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "status: str"]},
                    "headers": "original",
                }
            )
        )

        field_resolution = {"User ID": "user_id", "Status": "status"}
        sink.set_resume_field_resolution(field_resolution)

        result = sink.validate_output_target()
        assert result.valid, f"Validation failed: {result.error_message}"

    def test_jsonl_resume_validation_flexible_mode_with_original_headers(self, tmp_path: Path) -> None:
        """Flexible mode resume validation works with headers: original for JSONL."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # File has extra field not in schema (flexible mode allows this)
        with open(output_file, "w") as f:
            f.write(json.dumps({"User ID": "u1", "Amount": 100.0, "Extra Field": "extra"}) + "\n")

        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "flexible", "fields": ["user_id: str", "amount: float"]},
                    "headers": "original",
                }
            )
        )

        field_resolution = {"User ID": "user_id", "Amount": "amount"}
        sink.set_resume_field_resolution(field_resolution)

        result = sink.validate_output_target()
        assert result.valid, f"Validation failed: {result.error_message}"
