# tests/telemetry/exporters/test_console.py
"""Tests for Console telemetry exporter.

Tests cover:
- Configuration validation (format and output enum values)
- Type validation for configuration options
- Error handling
"""

import pytest

from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters.console import ConsoleExporter


class TestConsoleExporterConfiguration:
    """Tests for ConsoleExporter configuration."""

    def test_name_property(self) -> None:
        """Exporter name is 'console'."""
        exporter = ConsoleExporter()
        assert exporter.name == "console"

    def test_default_configuration(self) -> None:
        """Default configuration uses json format and stdout."""
        exporter = ConsoleExporter()
        exporter.configure({})
        assert exporter._format == "json"
        assert exporter._output == "stdout"

    def test_pretty_format(self) -> None:
        """Pretty format is valid."""
        exporter = ConsoleExporter()
        exporter.configure({"format": "pretty"})
        assert exporter._format == "pretty"

    def test_stderr_output(self) -> None:
        """stderr output is valid."""
        exporter = ConsoleExporter()
        exporter.configure({"output": "stderr"})
        assert exporter._output == "stderr"

    def test_invalid_format_raises(self) -> None:
        """Invalid format value raises TelemetryExporterError."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"format": "xml"})
        assert "Invalid format" in str(exc_info.value)

    def test_invalid_output_raises(self) -> None:
        """Invalid output value raises TelemetryExporterError."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"output": "file"})
        assert "Invalid output" in str(exc_info.value)

    def test_format_wrong_type_raises(self) -> None:
        """Non-string format raises TelemetryExporterError with clear message."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"format": 123})
        assert "'format' must be a string" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_output_wrong_type_raises(self) -> None:
        """Non-string output raises TelemetryExporterError with clear message."""
        exporter = ConsoleExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({"output": ["stdout", "stderr"]})
        assert "'output' must be a string" in str(exc_info.value)
        assert "list" in str(exc_info.value)


class TestConsoleExporterRegistration:
    """Tests for plugin registration."""

    def test_exporter_in_builtin_plugin(self) -> None:
        """ConsoleExporter is registered in BuiltinExportersPlugin."""
        from elspeth.telemetry.exporters import BuiltinExportersPlugin, ConsoleExporter

        plugin = BuiltinExportersPlugin()
        exporters = plugin.elspeth_get_exporters()
        assert ConsoleExporter in exporters

    def test_exporter_in_package_all(self) -> None:
        """ConsoleExporter is exported from package __all__."""
        from elspeth.telemetry import exporters

        assert "ConsoleExporter" in exporters.__all__
