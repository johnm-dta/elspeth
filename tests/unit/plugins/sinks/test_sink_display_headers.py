"""Tests for sink header output functionality.

Tests the unified headers configuration option for CSV and JSON sinks
that controls output header naming (normalized, original, or custom mapping).
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.plugin_context import PluginContext
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


class TestCSVSinkHeaders:
    """Tests for CSVSink header output functionality."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_explicit_custom_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """headers dict maps normalized names to display names in CSV header."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Transaction Amount"},
                }
            )
        )

        # Write with normalized field names
        sink.write(
            [
                {"user_id": "u1", "amount": 100.0},
                {"user_id": "u2", "amount": 200.0},
            ],
            ctx,
        )
        sink.flush()
        sink.close()

        # Verify CSV header uses display names
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["User ID", "Transaction Amount"]

            # Verify data rows are correct
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0] == ["u1", "100.0"]
            assert rows[1] == ["u2", "200.0"]

    def test_partial_custom_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Unmapped fields keep their normalized names."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float", "status: str"]},
                    "headers": {"user_id": "User ID"},  # Only user_id is mapped
                }
            )
        )

        sink.write([{"user_id": "u1", "amount": 100.0, "status": "active"}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            # user_id mapped, others keep normalized names
            assert header == ["User ID", "amount", "status"]

    def test_original_headers_from_landscape(self, tmp_path: Path) -> None:
        """headers: original fetches mapping from Landscape."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "case_study_1: str"]},
                    "headers": "original",
                }
            )
        )

        # Mock Landscape with field resolution
        mock_landscape = MagicMock()
        mock_landscape.get_source_field_resolution.return_value = {
            "User ID": "user_id",
            "case StUdY --- 1!!": "case_study_1",
        }

        ctx = make_context(landscape=mock_landscape)

        # on_start fetches the mapping
        sink.on_start(ctx)

        # Write with normalized field names
        sink.write([{"user_id": "u1", "case_study_1": "value"}], ctx)
        sink.flush()
        sink.close()

        # Verify CSV header uses original (pre-normalization) names
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["User ID", "case StUdY --- 1!!"]

    def test_original_headers_requires_landscape(self, tmp_path: Path) -> None:
        """headers: original fails if Landscape is not available.

        Note: Error occurs on first write(), not on_start(), because field resolution
        is only available after source iteration begins (lazy resolution).
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["id: str"]},
                    "headers": "original",
                }
            )
        )

        ctx = PluginContext(run_id="test-run", config={}, landscape=None)

        # on_start is now a no-op for original headers (lazy resolution)
        sink.on_start(ctx)

        # Error occurs on first write when resolution is attempted
        with pytest.raises(ValueError, match="requires Landscape"):
            sink.write([{"id": "test"}], ctx)

    def test_original_headers_requires_field_resolution(self, tmp_path: Path) -> None:
        """headers: original fails if source didn't record resolution.

        Note: Error occurs on first write(), not on_start(), because field resolution
        is only available after source iteration begins (lazy resolution).
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["id: str"]},
                    "headers": "original",
                }
            )
        )

        mock_landscape = MagicMock()
        mock_landscape.get_source_field_resolution.return_value = None

        ctx = make_context(landscape=mock_landscape)

        # on_start is now a no-op for original headers (lazy resolution)
        sink.on_start(ctx)

        # Error occurs on first write when resolution is attempted
        with pytest.raises(ValueError, match="did not record field resolution"):
            sink.write([{"id": "test"}], ctx)

    def test_transform_added_fields_use_normalized_names(self, tmp_path: Path) -> None:
        """Fields added by transforms use their normalized names (no original exists)."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "computed_score: float"]},
                    "headers": "original",
                }
            )
        )

        mock_landscape = MagicMock()
        # Source only had user_id, computed_score was added by transform
        mock_landscape.get_source_field_resolution.return_value = {
            "User ID": "user_id",
        }

        ctx = make_context(landscape=mock_landscape)
        sink.on_start(ctx)

        sink.write([{"user_id": "u1", "computed_score": 0.95}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            # user_id restored, computed_score keeps normalized name
            assert header == ["User ID", "computed_score"]

    def test_no_headers_option_uses_normalized(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Without headers option, headers are normalized field names."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                }
            )
        )

        sink.write([{"user_id": "u1", "amount": 100.0}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["user_id", "amount"]


class TestJSONSinkHeaders:
    """Tests for JSONSink header output functionality."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_explicit_custom_headers_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """headers dict maps normalized names to display names in JSONL keys."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "observed"},
                    "headers": {"user_id": "User ID", "amount": "Transaction Amount"},
                }
            )
        )

        sink.write(
            [
                {"user_id": "u1", "amount": 100.0},
                {"user_id": "u2", "amount": 200.0},
            ],
            ctx,
        )
        sink.flush()
        sink.close()

        # Verify JSONL uses display names as keys
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            row1 = json.loads(lines[0])
            assert row1 == {"User ID": "u1", "Transaction Amount": 100.0}

            row2 = json.loads(lines[1])
            assert row2 == {"User ID": "u2", "Transaction Amount": 200.0}

    def test_explicit_custom_headers_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """headers dict works with JSON array format."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "json",
                    "schema": {"mode": "observed"},
                    "headers": {"user_id": "User ID", "status": "Status"},
                }
            )
        )

        sink.write([{"user_id": "u1", "status": "active"}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0] == {"User ID": "u1", "Status": "active"}

    def test_original_headers_jsonl(self, tmp_path: Path) -> None:
        """headers: original fetches mapping from Landscape for JSONL."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "observed"},
                    "headers": "original",
                }
            )
        )

        mock_landscape = MagicMock()
        mock_landscape.get_source_field_resolution.return_value = {
            "User ID": "user_id",
            "Amount (USD)": "amount_usd",
        }

        ctx = make_context(landscape=mock_landscape)
        sink.on_start(ctx)

        sink.write([{"user_id": "u1", "amount_usd": 99.99}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            row = json.loads(f.readline())
            assert row == {"User ID": "u1", "Amount (USD)": 99.99}

    def test_no_headers_option_uses_normalized(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Without headers option, JSONL uses normalized field names."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "observed"},
                }
            )
        )

        sink.write([{"user_id": "u1", "amount": 100.0}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            row = json.loads(f.readline())
            assert row == {"user_id": "u1", "amount": 100.0}


class TestCSVCustomHeadersAppendMode:
    """Tests for CSV append mode with custom headers."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_append_with_custom_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode correctly validates and appends when custom headers match."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"

        # First write with custom headers
        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Amount"},
                }
            )
        )
        sink1.write([{"user_id": "u1", "amount": 100.0}], ctx)
        sink1.flush()
        sink1.close()

        # Append with same custom headers
        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Amount"},
                    "mode": "append",
                }
            )
        )
        sink2.write([{"user_id": "u2", "amount": 200.0}], ctx)
        sink2.flush()
        sink2.close()

        # Verify both rows present with correct headers
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["User ID", "Amount"]
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0] == ["u1", "100.0"]
            assert rows[1] == ["u2", "200.0"]


class TestCSVCustomHeadersSpecialCharacters:
    """Tests for CSV custom headers containing special characters."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_header_with_comma(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Custom headers containing commas are properly quoted in CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["amount: float", "currency: str"]},
                    "headers": {"amount": "Amount, USD", "currency": "Currency"},
                }
            )
        )

        sink.write([{"amount": 100.0, "currency": "USD"}], ctx)
        sink.flush()
        sink.close()

        # Read raw file to verify quoting
        with open(output_file) as f:
            content = f.read()
            # Header should be quoted because it contains comma
            assert '"Amount, USD"' in content or "'Amount, USD'" in content

        # Verify it parses correctly with csv module
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Amount, USD", "Currency"]
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0] == ["100.0", "USD"]

    def test_header_with_quotes(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Custom headers containing quotes are properly escaped in CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["value: str"]},
                    "headers": {"value": 'Value "quoted"'},
                }
            )
        )

        sink.write([{"value": "test"}], ctx)
        sink.flush()
        sink.close()

        # Verify it parses correctly with csv module
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ['Value "quoted"']

    def test_header_with_newline(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Custom headers containing newlines are properly quoted in CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["description: str"]},
                    "headers": {"description": "Description\n(multi-line)"},
                }
            )
        )

        sink.write([{"description": "test"}], ctx)
        sink.flush()
        sink.close()

        # Verify it parses correctly with csv module
        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Description\n(multi-line)"]


class TestJSONLCustomHeadersAppendMode:
    """Tests for JSONL append/resume mode with custom headers."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_append_with_custom_headers(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode correctly validates and appends when custom headers match."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"

        # First write with custom headers
        sink1 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Amount"},
                }
            )
        )
        sink1.write([{"user_id": "u1", "amount": 100.0}], ctx)
        sink1.flush()
        sink1.close()

        # Append with same custom headers
        sink2 = inject_write_failure(
            JSONSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "fixed", "fields": ["user_id: str", "amount: float"]},
                    "headers": {"user_id": "User ID", "amount": "Amount"},
                    "mode": "append",
                }
            )
        )
        sink2.write([{"user_id": "u2", "amount": 200.0}], ctx)
        sink2.flush()
        sink2.close()

        # Verify both rows present with display names as keys
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2
            row1 = json.loads(lines[0])
            row2 = json.loads(lines[1])
            assert row1 == {"User ID": "u1", "Amount": 100.0}
            assert row2 == {"User ID": "u2", "Amount": 200.0}

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

    def test_csv_resume_validation_without_resolution_skips(self, tmp_path: Path) -> None:
        """CSV resume validation skips when headers: original but no resolution provided.

        Without the field resolution mapping, comparing normalized schema names
        (user_id, amount_usd) against display names (User ID, Amount (USD)) would
        produce a misleading failure. The validation correctly skips and returns
        success — validation will be performed once headers are resolved during
        the first write() call.
        """
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

        # Without resolution, validation skips — can't compare normalized vs display names
        result = sink.validate_output_target()
        assert result.valid, "Should skip validation when resolution not yet available"

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
