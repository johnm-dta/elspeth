"""Tests for elspeth plugins commands."""

import pytest
from typer.testing import CliRunner

runner = CliRunner()


class TestPluginsListCommand:
    """Tests for plugins list command."""

    def test_plugins_list_shows_sources(self) -> None:
        """plugins list shows available sources."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()
        assert "json" in result.stdout.lower()

    def test_plugins_list_shows_sinks(self) -> None:
        """plugins list shows available sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "database" in result.stdout.lower()

    def test_plugins_list_has_sections(self) -> None:
        """plugins list organizes by type."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "source" in result.stdout.lower()
        assert "sink" in result.stdout.lower()

    def test_plugins_list_type_filter(self) -> None:
        """plugins list --type filters by plugin type."""
        from elspeth.cli import app

        # Filter to sources only
        result = runner.invoke(app, ["plugins", "list", "--type", "source"])
        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()
        # Should not show sinks
        assert "database" not in result.stdout.lower()

    def test_plugins_list_invalid_type(self) -> None:
        """plugins list --type with invalid type shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "invalid"])
        assert result.exit_code != 0
