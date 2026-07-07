"""Tests for elspeth plugins commands."""

import json
import subprocess
import sys

from typer.testing import CliRunner

runner = CliRunner()


class TestCliPluginCatalogAdapter:
    """Tests for the Typer-free plugin catalog adapter."""

    def test_list_plugins_payload_is_json_ready(self) -> None:
        """list_plugins_payload returns plain JSON-ready dictionaries."""
        from elspeth.cli_plugins import list_plugins_payload

        payload = list_plugins_payload(None)

        assert set(payload) == {"source", "transform", "sink"}
        assert any(plugin["name"] == "csv" for plugin in payload["source"])
        assert any(plugin["name"] == "passthrough" for plugin in payload["transform"])
        json.dumps(payload)

    def test_list_plugins_payload_type_filter(self) -> None:
        """list_plugins_payload honors plugin type filtering."""
        from elspeth.cli_plugins import list_plugins_payload

        payload = list_plugins_payload("source")

        assert set(payload) == {"source"}
        assert all(plugin["plugin_type"] == "source" for plugin in payload["source"])

    def test_inspect_plugin_payload_combines_schema_and_config_fields(self) -> None:
        """inspect payload joins schema detail with summary config fields."""
        from elspeth.cli_plugins import inspect_plugin_payload

        payload = inspect_plugin_payload("source", "csv")

        assert payload["name"] == "csv"
        assert payload["plugin_type"] == "source"
        assert isinstance(payload["description"], str)
        assert isinstance(payload["config_fields"], list)
        assert isinstance(payload["json_schema"], dict)
        assert isinstance(payload["knob_schema"], dict)
        json.dumps(payload)


class TestPluginsListCommand:
    """Tests for plugins list command."""

    def _get_section_content(self, output: str, section_header: str) -> str:
        """Extract content for a section by header.

        Args:
            output: Full CLI output
            section_header: Header like "SOURCES:" or "TRANSFORMS:"

        Returns:
            Content from header until next section or end
        """
        lines = output.split("\n")
        in_section = False
        section_lines = []
        for line in lines:
            if section_header in line.upper():
                in_section = True
                continue
            if in_section:
                # Check if we hit another section header
                if line.strip() and line.strip().endswith(":") and line.strip()[:-1].isupper():
                    break
                section_lines.append(line)
        return "\n".join(section_lines)

    def test_plugins_list_shows_sources_in_sources_section(self) -> None:
        """plugins list shows csv/json under SOURCES section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse sources section specifically
        sources_section = self._get_section_content(result.stdout, "SOURCES:")
        assert "csv" in sources_section.lower(), f"Expected 'csv' in SOURCES section, got: {sources_section}"
        assert "json" in sources_section.lower(), f"Expected 'json' in SOURCES section, got: {sources_section}"

    def test_plugins_list_shows_sinks_in_sinks_section(self) -> None:
        """plugins list shows database under SINKS section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse sinks section specifically
        sinks_section = self._get_section_content(result.stdout, "SINKS:")
        assert "database" in sinks_section.lower(), f"Expected 'database' in SINKS section, got: {sinks_section}"

    def test_plugins_list_shows_transforms_in_transforms_section(self) -> None:
        """plugins list shows transforms under TRANSFORMS section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse transforms section specifically
        transforms_section = self._get_section_content(result.stdout, "TRANSFORMS:")
        assert "passthrough" in transforms_section.lower(), f"Expected 'passthrough' in TRANSFORMS section, got: {transforms_section}"

    def test_plugins_list_has_all_sections(self) -> None:
        """plugins list organizes by type with section headers."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SOURCES:" in output_upper, "Missing SOURCES: section header"
        assert "TRANSFORMS:" in output_upper, "Missing TRANSFORMS: section header"
        assert "SINKS:" in output_upper, "Missing SINKS: section header"

    def test_plugins_list_type_filter_source(self) -> None:
        """plugins list --type source shows only sources."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "source"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SOURCES:" in output_upper, "Missing SOURCES: header"
        assert "SINKS:" not in output_upper, "Should not show SINKS: when filtering to source"
        assert "TRANSFORMS:" not in output_upper, "Should not show TRANSFORMS: when filtering to source"

        # csv should appear in sources section
        assert "csv" in result.stdout.lower()

    def test_plugins_list_type_filter_transform(self) -> None:
        """plugins list --type transform shows only transforms."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "transform"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "TRANSFORMS:" in output_upper, "Missing TRANSFORMS: header"
        assert "SOURCES:" not in output_upper, "Should not show SOURCES: when filtering to transform"
        assert "SINKS:" not in output_upper, "Should not show SINKS: when filtering to transform"

        # passthrough should appear in transforms section
        assert "passthrough" in result.stdout.lower()

    def test_plugins_list_type_filter_sink(self) -> None:
        """plugins list --type sink shows only sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "sink"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SINKS:" in output_upper, "Missing SINKS: header"
        assert "SOURCES:" not in output_upper, "Should not show SOURCES: when filtering to sink"
        assert "TRANSFORMS:" not in output_upper, "Should not show TRANSFORMS: when filtering to sink"

    def test_plugins_list_invalid_type_error_message(self) -> None:
        """plugins list --type with invalid type shows specific error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "invalid"])
        assert result.exit_code == 1, f"Expected exit code 1 for invalid type, got {result.exit_code}"
        # Should show the invalid type name
        output = result.stdout.lower() + (result.stderr or "").lower()
        assert "invalid" in output, f"Expected 'invalid' in error message, got: {output}"
        # Should mention valid types
        assert "valid types" in output or "source" in output, f"Expected mention of valid types, got: {output}"

    def test_plugins_list_json_output_is_parseable_and_pure_stdout(self) -> None:
        """plugins list --format json writes only parseable JSON to stdout."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--format", "json"])

        assert result.exit_code == 0
        assert result.stdout.startswith("{")
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert any(plugin["name"] == "csv" for plugin in payload["source"])
        assert "SOURCES:" not in result.stdout

    def test_plugins_list_json_type_filter(self) -> None:
        """plugins list --type source --format json emits only the filtered section."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "source", "--format", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload) == {"source"}
        assert all(plugin["plugin_type"] == "source" for plugin in payload["source"])

    def test_plugins_inspect_json_output(self) -> None:
        """plugins inspect exposes schema detail as JSON."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "inspect", "source", "csv", "--format", "json"])

        assert result.exit_code == 0
        assert result.stdout.startswith("{")
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        assert payload["name"] == "csv"
        assert payload["plugin_type"] == "source"
        assert isinstance(payload["config_fields"], list)
        assert isinstance(payload["json_schema"], dict)
        assert isinstance(payload["knob_schema"], dict)

    def test_plugins_inspect_text_output(self) -> None:
        """plugins inspect has a human-readable text mode."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "inspect", "transform", "passthrough"])

        assert result.exit_code == 0
        assert "passthrough" in result.stdout
        assert "transform" in result.stdout.lower()
        assert "JSON Schema" in result.stdout

    def test_plugins_inspect_invalid_plugin_exits_nonzero(self) -> None:
        """plugins inspect fails clearly for unknown plugins."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "inspect", "source", "does-not-exist", "--format", "json"])

        assert result.exit_code == 1
        output = result.stdout.lower() + (result.stderr or "").lower()
        assert "does-not-exist" in output

    def test_plugins_list_json_subprocess_stdout_stderr_separation(self) -> None:
        """Real subprocess output keeps JSON on stdout and success stderr empty."""
        completed = subprocess.run(
            [sys.executable, "-m", "elspeth.cli", "plugins", "list", "--format", "json"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0
        assert completed.stderr == ""
        assert completed.stdout.startswith("{")
        json.loads(completed.stdout)
