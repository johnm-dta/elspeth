"""Tests for elspeth explain command."""

from typer.testing import CliRunner

runner = CliRunner()


class TestExplainCommand:
    """Tests for explain command."""

    def test_explain_requires_run_id(self) -> None:
        """explain requires --run option."""
        from elspeth.cli import app

        result = runner.invoke(app, ["explain"])
        assert result.exit_code != 0
        assert (
            "missing" in result.output.lower() or "required" in result.output.lower()
        )

    def test_explain_no_tui_mode(self) -> None:
        """explain --no-tui outputs text instead of TUI."""
        from elspeth.cli import app

        # Note: This will fail gracefully since no runs exist
        result = runner.invoke(app, ["explain", "--run", "test-run", "--no-tui"])
        # Should not crash, may report "run not found"
        assert "error" in result.output.lower() or "not found" in result.output.lower()

    def test_explain_json_output(self) -> None:
        """explain --json outputs JSON format."""
        from elspeth.cli import app

        result = runner.invoke(app, ["explain", "--run", "test-run", "--json"])
        # Should output JSON (even if error)
        assert result.output.strip().startswith("{") or result.output.strip().startswith(
            "["
        )
