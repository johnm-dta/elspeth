# tests/unit/cli/test_cli.py
"""Tests for ELSPETH CLI basics.

Migrated from tests/cli/test_cli.py.
Tests that require LandscapeDB, Orchestrator, or real file I/O with
verify_audit_trail are deferred to integration tier.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_importable(self) -> None:
        """CLI app module can be imported without errors."""
        from elspeth.cli import app  # noqa: F401

    def test_version_flag(self) -> None:
        """--version shows version info."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "elspeth" in result.stdout.lower()

    def test_help_flag(self) -> None:
        """--help shows available commands."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "explain" in result.stdout
        assert "validate" in result.stdout
        assert "plugins" in result.stdout
        assert "resume" in result.stdout
        assert "purge" in result.stdout


class TestTildeExpansion:
    """Tests that CLI path options expand ~ to home directory.

    Regression tests for:
    - docs/bugs/closed/P2-2026-01-20-cli-paths-no-tilde-expansion.md
    """

    def test_run_expands_tilde_in_settings_path(self, tmp_path: Path) -> None:
        """run command expands ~ in --settings path.

        Creates a file in a temp dir, then constructs a path using ~ that
        resolves to the same location, verifying expansion works.
        """
        from elspeth.cli import app

        # Create a settings file
        settings_content = """
source:
  plugin: csv
  options:
    path: input.csv
    on_validation_failure: discard
    on_success: default
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        # Mock expanduser to return our temp path
        # This simulates ~ expanding to our test directory
        def mock_expanduser(self: Path) -> Path:
            if str(self).startswith("~"):
                return tmp_path / str(self)[2:]  # Replace ~/x with tmp_path/x
            return self

        with patch.object(Path, "expanduser", mock_expanduser):
            result = runner.invoke(app, ["run", "-s", "~/settings.yaml"])

        # Should find the file (even if validation fails for other reasons)
        # The key is that it doesn't say "file not found" for the tilde path
        assert "Settings file not found: ~/settings.yaml" not in result.output

    def test_validate_expands_tilde_in_settings_path(self, tmp_path: Path) -> None:
        """validate command expands ~ in --settings path."""
        from elspeth.cli import app

        settings_content = """
source:
  plugin: csv
  options:
    path: input.csv
    on_validation_failure: discard
    on_success: default
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        def mock_expanduser(self: Path) -> Path:
            if str(self).startswith("~"):
                return tmp_path / str(self)[2:]
            return self

        with patch.object(Path, "expanduser", mock_expanduser):
            result = runner.invoke(app, ["validate", "-s", "~/settings.yaml"])

        # Should find the file
        assert "Settings file not found: ~/settings.yaml" not in result.output


class TestResumeDbValidation:
    """Regression tests for resume database validation.

    Regression test for ti23: resume with settings-derived database URL
    that points to a non-existent SQLite file should fail with a clear
    "Database file not found" error, not silently create an empty DB
    and report "run not found".
    """

    def test_resume_rejects_nonexistent_settings_db(self, tmp_path: Path) -> None:
        """resume exits with error when settings landscape.url points to missing file."""
        from elspeth.cli import app

        # Create a settings file whose landscape.url points to a non-existent DB
        nonexistent_db = tmp_path / "does_not_exist.db"
        settings_content = f"""
source:
  plugin: csv
  on_success: default
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
landscape:
  url: "sqlite:///{nonexistent_db}"
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        result = runner.invoke(app, ["resume", "fake-run-id", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit code 1, got {result.exit_code}. Output: {result.output}"
        assert "Database file not found" in result.output, f"Expected 'Database file not found' error, got: {result.output}"
        # Must NOT have created the file
        assert not nonexistent_db.exists(), "resume should not create a new database file"

    def test_resume_rejects_non_landscape_db(self, tmp_path: Path) -> None:
        """resume exits cleanly when DB exists but has no Landscape tables.

        Regression test: switching to create_tables=False caused an uncaught
        OperationalError ('no such table: runs') when the DB file exists but
        is not a Landscape database. Must give a clear CLI error instead.
        """
        from elspeth.cli import app

        # Create an empty SQLite file (exists but has no tables)
        empty_db = tmp_path / "not_landscape.db"
        import sqlite3

        conn = sqlite3.connect(str(empty_db))
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.close()

        settings_content = f"""
source:
  plugin: csv
  on_success: default
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
landscape:
  url: "sqlite:///{empty_db}"
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        result = runner.invoke(app, ["resume", "fake-run-id", "-s", str(settings_file)])

        assert result.exit_code == 1, f"Expected exit code 1, got {result.exit_code}. Output: {result.output}"
        assert "does not appear to be an ELSPETH audit database" in result.output, (
            f"Expected clear error about missing tables, got: {result.output}"
        )
        # Must NOT have crashed with a traceback
        assert "Traceback" not in result.output, f"Should not show traceback: {result.output}"


class TestExplainPassphraseResolution:
    """T4: Settings loading failure during passphrase resolution must not be silent.

    When --settings is explicitly provided and the YAML is malformed,
    explain should exit with an error, not silently continue with
    passphrase=None.
    """

    def test_explain_exits_on_malformed_settings_yaml(self, tmp_path: Path) -> None:
        """explain --settings with malformed YAML exits with code 1."""
        from elspeth.cli import app
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape import LandscapeDB
        from tests.fixtures.landscape import make_factory

        # Create a valid Landscape database with one run
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.run_lifecycle.complete_run("run-1", RunStatus.COMPLETED)
        db.close()

        # Create a malformed YAML settings file
        bad_settings = tmp_path / "bad_settings.yaml"
        bad_settings.write_text("source:\n  plugin: csv\n  options: [\ninvalid yaml")

        result = runner.invoke(
            app,
            [
                "explain",
                "--database",
                str(db_path),
                "--settings",
                str(bad_settings),
                "--run",
                "run-1",
                "--row",
                "row-1",
                "--no-tui",
            ],
        )

        assert result.exit_code == 1, f"Expected exit code 1 for malformed settings YAML, got {result.exit_code}. Output: {result.output}"
        # Error message should mention the YAML/settings problem
        output_lower = result.output.lower()
        assert "yaml" in output_lower or "syntax" in output_lower or "settings" in output_lower, (
            f"Expected error about YAML/settings, got: {result.output}"
        )

    def test_explain_succeeds_without_settings(self, tmp_path: Path) -> None:
        """explain --database without --settings should work (passphrase=None)."""
        from elspeth.cli import app
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape import LandscapeDB
        from tests.fixtures.landscape import make_factory

        # Create a valid Landscape database with one run
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.run_lifecycle.complete_run("run-1", RunStatus.COMPLETED)
        db.close()

        result = runner.invoke(
            app,
            [
                "explain",
                "--database",
                str(db_path),
                "--run",
                "run-1",
                "--no-tui",
            ],
        )

        # Without --settings, passphrase fallthrough to None is correct
        # and the command should not fail on settings loading
        assert "yaml" not in result.output.lower() or result.exit_code == 0


class TestExplainSecretLoading:
    """explain must use _load_settings_with_secrets, not load_settings.

    Bug: elspeth-866ccf203b — explain bypassed the secret-loading flow
    when reloading --settings for passphrase resolution. Secrets were
    never injected, so encrypted audit databases became unreadable.
    """

    def test_explain_uses_secret_loading_path(self, tmp_path: Path) -> None:
        """explain --settings should call _load_settings_with_secrets, not load_settings."""
        from unittest.mock import MagicMock

        from elspeth.cli import app
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape import LandscapeDB
        from tests.fixtures.landscape import make_factory

        # Create a valid Landscape database with one run
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.run_lifecycle.complete_run("run-1", RunStatus.COMPLETED)
        db.close()

        # Create a valid settings file
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("source:\n  plugin: csv\n  options:\n    file: test.csv\n")

        # Patch _load_settings_with_secrets to verify it's called
        mock_settings = MagicMock()
        mock_settings.landscape = None
        with patch(
            "elspeth.cli._load_settings_with_secrets",
            return_value=(mock_settings, []),
        ) as mock_load:
            runner.invoke(
                app,
                [
                    "explain",
                    "--database",
                    str(db_path),
                    "--settings",
                    str(settings_file),
                    "--run",
                    "run-1",
                    "--no-tui",
                ],
            )
            mock_load.assert_called_once()


class TestBuildResumeGraphs:
    """Test _build_resume_graphs accepts connection-valued on_success.

    Regression test for 6v1d: resume mode must accept connection names (e.g.
    'source_out') not just sink names as source.on_success. The previous
    implementation rejected connection-valued on_success with a typer.Exit(1).
    """

    def test_connection_valued_on_success_accepted(self, plugin_manager) -> None:
        """_build_resume_graphs succeeds when source.on_success is a connection name."""
        from elspeth.cli import _build_resume_graphs
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            transforms=[
                TransformSettings(
                    name="processor",
                    plugin="passthrough",
                    input="source_out",
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(config)
        validation_graph, execution_graph = _build_resume_graphs(config, plugins)

        # Both graphs should build successfully
        assert validation_graph.node_count > 0
        assert execution_graph.node_count > 0

    def test_sink_valued_on_success_still_accepted(self, plugin_manager) -> None:
        """_build_resume_graphs still works when source.on_success is a sink name."""
        from elspeth.cli import _build_resume_graphs
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings

        config = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="output",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(config)
        validation_graph, execution_graph = _build_resume_graphs(config, plugins)

        assert validation_graph.node_count > 0
        assert execution_graph.node_count > 0


class TestHealthCommand:
    """Tests for the health command.

    The health command checks system health for deployment verification.
    Design invariants:
    - "ok" = check passed
    - "warn" = degraded but functional (does NOT fail health check)
    - "error" = check failed (fails health check, exit code 1)
    - "skip" = check not applicable

    Warn-level checks (informational, don't affect exit code):
    - git commit SHA unavailable
    - config directory not found
    - output directory not found/writable

    Error-level checks (fail health check, exit code 1):
    - web interface not reachable
    - database connection failed
    - plugins failed to load
    """

    def test_health_returns_json_output(self) -> None:
        """health --json returns valid JSON with expected structure."""
        import json

        from elspeth.cli import app

        # Web check skipped by default, so this tests basic health structure
        result = runner.invoke(app, ["health", "--json"])

        # Parse the JSON output
        data = json.loads(result.stdout)
        assert "status" in data
        assert "version" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_health_fails_when_web_not_running(self, monkeypatch) -> None:
        """health --check-web exits 1 when web interface is not reachable.

        This is a critical deployment check: the web server MUST be running
        when --check-web is specified (web containers).
        """
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Mock build_opener to return a mock opener that raises URLError on .open()
        # The implementation uses build_opener(...).open(...), not urlopen directly,
        # to bypass HTTP_PROXY in corporate environments.
        mock_opener = MagicMock()
        mock_opener.open.side_effect = urllib.error.URLError("connection refused")

        def mock_build_opener(*args, **kwargs):
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # --check-web required since web checking is now opt-in (default: skip)
        result = runner.invoke(app, ["health", "--check-web", "--json"])

        assert result.exit_code == 1, f"Expected exit 1 when web is down, got {result.exit_code}"

        import json

        data = json.loads(result.stdout)
        assert data["status"] == "unhealthy"
        assert data["checks"]["web"]["status"] == "error"
        assert "connection failed" in data["checks"]["web"]["value"]

    def test_health_warn_does_not_fail_check(self) -> None:
        """warn-level checks do not cause health check failure.

        The health command treats 'warn' as 'degraded but functional'.
        Missing git SHA, config dir, or output dir are warnings, not errors.
        These don't indicate the system is broken - just informational.
        """
        import json

        from elspeth.cli import app

        # With web skipped by default, basic health check should pass
        result = runner.invoke(app, ["health", "--json"])
        data = json.loads(result.stdout)

        # Verify git check exists and uses warn (not error) when unavailable
        assert "commit" in data["checks"]

        # Warn-level checks should never appear as errors
        error_checks = [name for name, info in data["checks"].items() if info["status"] == "error"]
        assert "commit" not in error_checks, "git commit should be warn, not error"
        assert "config_dir" not in error_checks, "config_dir should be warn, not error"
        assert "output_dir" not in error_checks, "output_dir should be warn, not error"

    def test_health_verbose_shows_check_details(self) -> None:
        """health --verbose shows detailed check information."""
        from elspeth.cli import app

        result = runner.invoke(app, ["health", "--verbose"])

        # Should show check names and values
        assert "Checks:" in result.stdout
        assert "version" in result.stdout.lower()
        assert "web" in result.stdout.lower()

    def test_health_with_web_running(self, monkeypatch) -> None:
        """health exits 0 when web server returns ok.

        This tests the happy path by mocking the no-proxy opener.
        """
        import json
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Mock build_opener to return a mock opener that returns success
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = lambda self: self
        mock_response.__exit__ = lambda self, *args: None

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_response
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        result = runner.invoke(app, ["health", "--check-web", "--json"])
        data = json.loads(result.stdout)

        assert data["checks"]["web"]["status"] == "ok"
        # Note: exit code may still be 1 if other checks fail (e.g., plugins)

    def test_health_web_check_uses_env_vars(self, monkeypatch) -> None:
        """web check reads host/port from ELSPETH_WEB__* env vars."""
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Set custom host/port
        monkeypatch.setenv("ELSPETH_WEB__HOST", "custom-host")
        monkeypatch.setenv("ELSPETH_WEB__PORT", "9999")

        # Capture which URL was requested
        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        runner.invoke(app, ["health", "--check-web", "--json"])

        assert len(requested_urls) == 1
        assert "custom-host:9999" in requested_urls[0]

    def test_health_skip_web_option(self) -> None:
        """--skip-web skips web check for batch pipeline containers."""
        import json

        from elspeth.cli import app

        result = runner.invoke(app, ["health", "--skip-web", "--json"])
        data = json.loads(result.stdout)

        # Web check should be skipped, not failed
        assert data["checks"]["web"]["status"] == "skip"
        assert "skipped via --skip-web" in data["checks"]["web"]["value"]

        # Exit code should be 0 if no other checks fail
        # (plugins might fail in test environment, so check web specifically)
        assert data["checks"]["web"]["status"] != "error"

    def test_health_port_option_overrides_env(self, monkeypatch) -> None:
        """--port takes precedence over ELSPETH_WEB__PORT env var."""
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Set env var to wrong port
        monkeypatch.setenv("ELSPETH_WEB__PORT", "9999")

        # Capture which URL was requested
        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # CLI flag should override env var (--check-web required for web probe)
        runner.invoke(app, ["health", "--check-web", "--port", "7777", "--json"])

        assert len(requested_urls) == 1
        assert ":7777/" in requested_urls[0], f"Expected port 7777, got {requested_urls[0]}"
        assert ":9999/" not in requested_urls[0], "Env var port should be overridden"

    def test_health_host_option_overrides_env(self, monkeypatch) -> None:
        """--host takes precedence over ELSPETH_WEB__HOST env var."""
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Set env var to wrong host
        monkeypatch.setenv("ELSPETH_WEB__HOST", "wrong-host")

        # Capture which URL was requested
        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # CLI flag should override env var (--check-web required for web probe)
        runner.invoke(app, ["health", "--check-web", "--host", "correct-host", "--json"])

        assert len(requested_urls) == 1
        assert "correct-host:" in requested_urls[0]
        assert "wrong-host" not in requested_urls[0]

    def test_health_port_implies_check_web(self, monkeypatch) -> None:
        """--port alone implies --check-web.

        When user provides --port, they clearly want to check the web server.
        Requiring explicit --check-web is a footgun that causes false-positive
        health checks in web deployments.
        """
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # --port alone should trigger web check (no --check-web needed)
        runner.invoke(app, ["health", "--port", "9999", "--json"])

        assert len(requested_urls) == 1, "Expected web probe when --port is provided"
        assert ":9999/" in requested_urls[0], f"Expected port 9999, got {requested_urls[0]}"

    def test_health_host_implies_check_web(self, monkeypatch) -> None:
        """--host alone implies --check-web.

        When user provides --host, they clearly want to check the web server.
        Requiring explicit --check-web is a footgun that causes false-positive
        health checks in web deployments.
        """
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # --host alone should trigger web check (no --check-web needed)
        runner.invoke(app, ["health", "--host", "my-web-server", "--json"])

        assert len(requested_urls) == 1, "Expected web probe when --host is provided"
        assert "my-web-server:" in requested_urls[0], f"Expected host my-web-server, got {requested_urls[0]}"

    def test_health_explicit_skip_web_honored_with_host_port(self, monkeypatch) -> None:
        """Explicit --skip-web is honored even when --host/--port are supplied.

        This supports wrappers that always pass host/port values but only
        sometimes want to suppress the web check. The inference from host/port
        only applies when skip_web is at its default value.
        """
        import json
        from unittest.mock import MagicMock

        from elspeth.cli import app

        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise Exception("Should not be called")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # --skip-web should be honored even when --host is provided
        result = runner.invoke(app, ["health", "--skip-web", "--host", "example.com", "--json"])

        # Web check should be skipped — no HTTP request made
        assert len(requested_urls) == 0, f"Expected no web probe with explicit --skip-web, got {requested_urls}"

        data = json.loads(result.stdout)
        assert data["checks"]["web"]["status"] == "skip"
        assert "skipped" in data["checks"]["web"]["value"].lower()

    def test_health_ipv6_host_is_bracketed(self, monkeypatch) -> None:
        """IPv6 hosts must be bracketed in URLs per RFC 3986.

        Without brackets, http://::1:8451/api/health is ambiguous —
        urllib can't distinguish port from IPv6 address components.
        """
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # Use IPv6 loopback (--check-web required for web probe)
        runner.invoke(app, ["health", "--check-web", "--host", "::1", "--json"])

        assert len(requested_urls) == 1
        # IPv6 must be bracketed: http://[::1]:8451/api/health
        assert "[::1]:" in requested_urls[0], f"IPv6 host not bracketed: {requested_urls[0]}"

    def test_health_malformed_port_env_returns_unhealthy(self, monkeypatch) -> None:
        """Malformed ELSPETH_WEB__PORT should produce unhealthy result, not crash.

        A deployment health probe needs to distinguish "misconfigured" from
        "CLI binary crashed". Crashing on bad config is a regression.
        """
        import json

        from elspeth.cli import app

        # Set port to non-numeric value
        monkeypatch.setenv("ELSPETH_WEB__PORT", "not-a-number")

        # --check-web required to exercise the malformed port path
        result = runner.invoke(app, ["health", "--check-web", "--json"])

        # Command must complete without unexpected exception.
        # SystemExit(1) is expected for unhealthy status — it's an intentional
        # exit, not a crash. Actual crashes would be ValueError, TypeError, etc.
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise AssertionError(f"CLI crashed on malformed port: {result.exception}")

        # Must produce valid JSON (not a stack trace)
        data = json.loads(result.stdout)
        # Must report as unhealthy with meaningful error
        assert data["status"] == "unhealthy"
        assert data["checks"]["web"]["status"] == "error"
        # Error message should indicate the bad port value
        assert "int()" in data["checks"]["web"]["value"] or "not-a-number" in data["checks"]["web"]["value"]

    def test_health_wildcard_host_translated_to_loopback(self, monkeypatch) -> None:
        """Wildcard bind addresses must be translated to loopback for probing.

        0.0.0.0 and :: are listen-all addresses, not connectable destinations.
        A web server bound to 0.0.0.0:8451 is reachable at 127.0.0.1:8451,
        not at 0.0.0.0:8451.
        """
        import urllib.error
        from unittest.mock import MagicMock

        from elspeth.cli import app

        requested_urls: list[str] = []

        def mock_build_opener(*handlers):
            mock_opener = MagicMock()

            def mock_open(url, timeout=None):
                requested_urls.append(url)
                raise urllib.error.URLError("connection refused")

            mock_opener.open.side_effect = mock_open
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", mock_build_opener)

        # Test IPv4 wildcard (--check-web required for web probe)
        runner.invoke(app, ["health", "--check-web", "--host", "0.0.0.0", "--json"])
        assert len(requested_urls) == 1
        assert "127.0.0.1:" in requested_urls[0], f"0.0.0.0 should be translated to 127.0.0.1, got {requested_urls[0]}"
        assert "0.0.0.0" not in requested_urls[0], f"Wildcard 0.0.0.0 should not appear in probe URL: {requested_urls[0]}"

        # Test IPv6 wildcard
        requested_urls.clear()
        runner.invoke(app, ["health", "--check-web", "--host", "::", "--json"])
        assert len(requested_urls) == 1
        assert "[::1]:" in requested_urls[0], f":: should be translated to [::1], got {requested_urls[0]}"
        # :: has brackets so would appear as [::] if not translated
        assert "[::]" not in requested_urls[0], f"Wildcard :: should not appear in probe URL: {requested_urls[0]}"

    def test_health_web_check_bypasses_proxy_env_vars(self, monkeypatch) -> None:
        """Health probe must bypass HTTP_PROXY/HTTPS_PROXY for local self-checks.

        Corporate environments often set HTTP_PROXY for outbound traffic. If the
        health check respects these variables, it routes localhost probes through
        the proxy, causing false unhealthy reports when the proxy can't reach the
        local ELSPETH web process.

        The fix: use urllib.request.build_opener with ProxyHandler({}) to create
        a direct-connect opener that ignores proxy environment variables.
        """
        import urllib.request
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Set proxy env vars that would normally affect urllib
        monkeypatch.setenv("HTTP_PROXY", "http://corporate-proxy.example.com:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://corporate-proxy.example.com:8080")

        # Track what handlers are passed to build_opener
        captured_handlers: list[Any] = []

        def capturing_build_opener(*handlers):
            captured_handlers.extend(handlers)
            # Return a mock opener that simulates success
            mock_opener = MagicMock()
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "ok"}'
            mock_response.__enter__ = lambda self: self
            mock_response.__exit__ = lambda self, *args: None
            mock_opener.open.return_value = mock_response
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", capturing_build_opener)

        # --check-web required to exercise the proxy bypass code path
        runner.invoke(app, ["health", "--check-web", "--json"])

        # Verify we created a custom opener with ProxyHandler({})
        assert len(captured_handlers) >= 1, "Expected build_opener to be called with handlers"

        proxy_handlers = [h for h in captured_handlers if isinstance(h, urllib.request.ProxyHandler)]
        assert len(proxy_handlers) == 1, f"Expected exactly one ProxyHandler, got {len(proxy_handlers)}"

        # ProxyHandler({}) creates a handler with no proxies configured
        # This bypasses the environment variable proxies
        handler = proxy_handlers[0]
        assert handler.proxies == {}, f"ProxyHandler should have empty proxies dict, got {handler.proxies}"  # type: ignore[attr-defined]  # proxies is set dynamically by ProxyHandler.__init__

    def test_health_remote_host_honors_proxy_env_vars(self, monkeypatch) -> None:
        """Health probe must honor HTTP_PROXY for remote (non-loopback) hosts.

        When probing a remote ELSPETH instance via --host, the check must use
        the configured proxy if one is set. Only loopback self-checks bypass
        proxies; remote checks in proxied environments need the proxy to reach
        the target.
        """
        import urllib.request
        from unittest.mock import MagicMock

        from elspeth.cli import app

        # Set proxy env vars
        monkeypatch.setenv("HTTP_PROXY", "http://corporate-proxy.example.com:8080")

        # Track what handlers are passed to build_opener
        captured_handlers: list[Any] = []

        def capturing_build_opener(*handlers):
            captured_handlers.extend(handlers)
            mock_opener = MagicMock()
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "ok"}'
            mock_response.__enter__ = lambda self: self
            mock_response.__exit__ = lambda self, *args: None
            mock_opener.open.return_value = mock_response
            return mock_opener

        monkeypatch.setattr("urllib.request.build_opener", capturing_build_opener)

        # Probe a remote host — should NOT bypass proxy
        runner.invoke(app, ["health", "--host", "remote-elspeth.example.com", "--json"])

        # For remote hosts, build_opener should be called with NO ProxyHandler
        # (allowing urllib to use its default proxy-from-environment behavior)
        proxy_handlers = [h for h in captured_handlers if isinstance(h, urllib.request.ProxyHandler)]
        assert len(proxy_handlers) == 0, (
            f"Expected no ProxyHandler for remote host (should honor HTTP_PROXY), but got {len(proxy_handlers)}"
        )


class TestResumeErrorPaths:
    """Tests for resume command error handling edge cases.

    These tests verify that the resume command fails cleanly with informative
    error messages for various failure modes, rather than crashing with tracebacks.

    Error paths tested:
    - Secret loading failures (Key Vault, env vars)
    - Passphrase resolution failures (SQLCipher)
    - Payload store configuration errors
    - Sink resume compatibility checks
    - Schema validation failures
    """

    def _make_settings_with_landscape_db(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create settings file and a valid Landscape DB for testing resume errors.

        Returns (settings_path, db_path) where db_path has Landscape schema.
        Most tests mock before database access, so we only need schema, not data.
        """
        from elspeth.core.landscape import LandscapeDB

        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        db.close()

        settings_content = f"""
source:
  plugin: csv
  on_success: default
  options:
    path: input.csv
    on_validation_failure: discard
    schema:
      mode: observed
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: {tmp_path / "output.json"}
      schema:
        mode: observed
landscape:
  url: "sqlite:///{db_path}"
payload_store:
  backend: filesystem
  base_path: {tmp_path / "payloads"}
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)
        return settings_file, db_path

    def test_resume_rejects_secret_load_error(self, tmp_path: Path) -> None:
        """resume exits cleanly when secrets fail to load.

        SecretLoadError can occur when Key Vault is configured but unavailable,
        or when required environment variables are missing.
        """
        from unittest.mock import patch

        from elspeth.cli import app
        from elspeth.core.security.config_secrets import SecretLoadError

        settings_file, _db_path = self._make_settings_with_landscape_db(tmp_path)

        with patch(
            "elspeth.cli._load_settings_with_secrets",
            side_effect=SecretLoadError("Key Vault authentication failed"),
        ):
            result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file)])

        assert result.exit_code == 1
        assert "Error loading secrets" in result.output
        assert "Key Vault authentication failed" in result.output
        assert "Traceback" not in result.output

    def test_resume_rejects_passphrase_resolution_error(self, tmp_path: Path) -> None:
        """resume exits cleanly when SQLCipher passphrase cannot be resolved.

        Passphrase resolution can fail if the configured source (env var, file)
        is not available.
        """
        from unittest.mock import patch

        from elspeth.cli import app

        settings_file, _db_path = self._make_settings_with_landscape_db(tmp_path)

        with patch(
            "elspeth.cli_helpers.resolve_audit_passphrase",
            side_effect=RuntimeError("Passphrase not found in AUDIT_PASSPHRASE env var"),
        ):
            result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file)])

        assert result.exit_code == 1
        assert "Error:" in result.output
        assert "Passphrase" in result.output
        assert "Traceback" not in result.output

    def test_resume_rejects_pydantic_validation_error(self, tmp_path: Path) -> None:
        """resume exits cleanly with structured error when config fails validation.

        Pydantic ValidationError should be caught and formatted as user-friendly
        error messages, not raw exception text.
        """
        from elspeth.cli import app

        # Create settings with invalid config (missing required field)
        settings_content = """
source:
  plugin: csv
  # Missing on_success
  options:
    path: input.csv
    on_validation_failure: discard
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file)])

        assert result.exit_code == 1
        # Should show structured error, not ValidationError traceback
        output_lower = result.output.lower()
        assert "error" in output_lower
        # Should mention the missing field
        assert "on_success" in output_lower or "configuration" in output_lower
        assert "Traceback" not in result.output

    def test_resume_rejects_cannot_resume_check(self, tmp_path: Path) -> None:
        """resume exits cleanly when RecoveryManager.can_resume returns False.

        This happens when:
        - Run is completed (nothing to resume)
        - Run is still running (concurrent resume)
        - Checkpoint is incompatible (topology changed)
        """
        from unittest.mock import patch

        from elspeth.cli import app
        from elspeth.contracts.checkpoint import ResumeCheck

        settings_file, _db_path = self._make_settings_with_landscape_db(tmp_path)

        # Mock can_resume to return False with a reason
        mock_check = ResumeCheck(can_resume=False, reason="Run already completed successfully")

        with patch("elspeth.core.checkpoint.RecoveryManager") as MockRecovery:
            MockRecovery.return_value.can_resume.return_value = mock_check
            result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file)])

        assert result.exit_code == 1
        assert "Cannot resume run" in result.output
        assert "already completed" in result.output
        assert "Traceback" not in result.output

    def test_resume_rejects_missing_payload_directory(self, tmp_path: Path) -> None:
        """resume --execute exits cleanly when payload directory doesn't exist.

        The payload store path must exist for resume to work - payloads contain
        the original row data needed to replay processing.
        """
        from unittest.mock import MagicMock, patch

        from elspeth.cli import app
        from elspeth.contracts.checkpoint import ResumeCheck, ResumePoint

        settings_file, _db_path = self._make_settings_with_landscape_db(tmp_path)
        # Do NOT create the payload directory - this is the error condition
        # (The settings file references tmp_path / "payloads" which doesn't exist)

        # Mock recovery manager to pass validation
        mock_check = ResumeCheck(can_resume=True)
        mock_resume_point = MagicMock(spec=ResumePoint)
        mock_resume_point.token_id = "tok-1"
        mock_resume_point.node_id = "node-1"
        mock_resume_point.sequence_number = 0
        mock_resume_point.aggregation_state = None
        mock_resume_point.coalesce_state = None

        with patch("elspeth.core.checkpoint.RecoveryManager") as MockRecovery:
            MockRecovery.return_value.can_resume.return_value = mock_check
            MockRecovery.return_value.get_resume_point.return_value = mock_resume_point
            MockRecovery.return_value.get_unprocessed_rows.return_value = ["row-1"]
            result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file), "--execute"])

        assert result.exit_code == 1
        assert "Payload directory not found" in result.output
        assert "Traceback" not in result.output

    def test_resume_rejects_unsupported_payload_backend(self, tmp_path: Path) -> None:
        """resume --execute exits cleanly for non-filesystem payload backends.

        Currently only 'filesystem' backend is supported for resume.
        """
        from unittest.mock import MagicMock, patch

        from elspeth.cli import app
        from elspeth.contracts.checkpoint import ResumeCheck, ResumePoint

        # Create settings with unsupported backend
        db_path = tmp_path / "audit.db"
        settings_content = f"""
source:
  plugin: csv
  on_success: default
  options:
    path: input.csv
    on_validation_failure: discard
    schema:
      mode: observed
sinks:
  default:
    plugin: json
    on_write_failure: discard
    options:
      path: output.json
      schema:
        mode: observed
landscape:
  url: "sqlite:///{db_path}"
payload_store:
  backend: azure_blob
  base_path: container/path
"""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(settings_content)

        # Create a valid Landscape DB (schema only - RecoveryManager is mocked)
        from elspeth.core.landscape import LandscapeDB

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        db.close()

        # Mock recovery manager to pass validation
        mock_check = ResumeCheck(can_resume=True)
        mock_resume_point = MagicMock(spec=ResumePoint)
        mock_resume_point.token_id = "tok-1"
        mock_resume_point.node_id = "node-1"
        mock_resume_point.sequence_number = 0
        mock_resume_point.aggregation_state = None
        mock_resume_point.coalesce_state = None

        with patch("elspeth.core.checkpoint.RecoveryManager") as MockRecovery:
            MockRecovery.return_value.can_resume.return_value = mock_check
            MockRecovery.return_value.get_resume_point.return_value = mock_resume_point
            MockRecovery.return_value.get_unprocessed_rows.return_value = ["row-1"]
            result = runner.invoke(app, ["resume", "test-run-123", "-s", str(settings_file), "--execute"])

        assert result.exit_code == 1
        assert "Unsupported payload store backend" in result.output
        assert "azure_blob" in result.output
        assert "filesystem" in result.output
        assert "Traceback" not in result.output
