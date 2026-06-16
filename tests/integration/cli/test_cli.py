# tests/integration/cli/test_cli.py
"""Integration tests for CLI end-to-end workflow."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import insert
from typer.testing import CliRunner

# Note: In Click 8.0+, mix_stderr is no longer a CliRunner parameter.
# Stderr output is combined with stdout by default when using CliRunner.invoke()
runner = CliRunner()

# ---------------------------------------------------------------------------
# Module-level constants (used by TestJoinCommand helpers)
# ---------------------------------------------------------------------------
_JOIN_NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
_JOIN_WINDOW = 80.0


class TestCLIIntegration:
    """End-to-end CLI integration tests."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95\n2,bob,87\n3,carol,92\n")
        return csv_file

    @pytest.fixture
    def pipeline_config(self, tmp_path: Path, sample_csv: Path) -> Path:
        """Create pipeline configuration.

        Note: Uses "default" as primary sink - the Orchestrator routes
        all completed rows to the "default" sink via output_sink.
        """
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(sample_csv),
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                }
            },
            "sinks": {
                # "default" is required - Orchestrator routes completed rows here
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            # Use temp-path DB to avoid polluting CWD during tests
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_full_workflow_csv_to_json(self, pipeline_config: Path, tmp_path: Path) -> None:
        """Complete workflow: validate, run with --execute, check output."""
        from elspeth.cli import app

        # Step 1: Validate configuration
        result = runner.invoke(app, ["validate", "-s", str(pipeline_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

        # Step 2: Run pipeline with --execute flag (required for safety)
        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--execute"])
        assert result.exit_code == 0
        assert "completed" in result.stdout.lower()

        # Step 3: Check output exists and is valid
        output_file = tmp_path / "output.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data) == 3
        assert data[0]["name"] == "alice"

    def test_plugins_list_shows_all_types(self) -> None:
        """plugins list shows sources and sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Sources
        assert "csv" in result.stdout
        assert "json" in result.stdout

        # Sinks
        assert "database" in result.stdout

    def test_dry_run_does_not_create_output(self, pipeline_config: Path, tmp_path: Path) -> None:
        """dry-run does not create output files."""
        from elspeth.cli import app

        output_file = tmp_path / "output.json"
        assert not output_file.exists()

        result = runner.invoke(app, ["run", "-s", str(pipeline_config), "--dry-run"])
        assert result.exit_code == 0

        # Output should NOT be created
        assert not output_file.exists()

    def test_run_without_flags_exits_with_warning(self, pipeline_config: Path) -> None:
        """run without --execute shows warning and exits non-zero."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(pipeline_config)])

        # Should exit with code 1 (safety feature)
        assert result.exit_code == 1
        # Should tell user to add --execute flag (in stderr, captured in output)
        assert "--execute" in result.output


class TestSourceQuarantineRouting:
    """Integration tests for source quarantine routing to sinks.

    Verifies that invalid source rows with on_validation_failure configured
    to a sink name are actually routed to that sink (not silently dropped).
    """

    @pytest.fixture
    def csv_with_invalid_rows(self, tmp_path: Path) -> Path:
        """Create CSV with mixed valid and invalid rows.

        Uses strict schema requiring id:int, name:str, score:int.
        Row 2 has score='bad' which fails int validation.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,alice,95\n"
            "2,bob,bad\n"  # Invalid: score is not an int
            "3,carol,92\n"
        )
        return csv_file

    @pytest.fixture
    def quarantine_pipeline_config(self, tmp_path: Path, csv_with_invalid_rows: Path) -> Path:
        """Create pipeline with quarantine sink for invalid rows."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(csv_with_invalid_rows),
                        "on_validation_failure": "quarantine",  # Route to quarantine sink
                        "schema": {
                            "mode": "fixed",
                            "fields": ["id: int", "name: str", "score: int"],
                        },
                    },
                }
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
                "quarantine": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "quarantine.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_invalid_rows_routed_failure_to_quarantine_sink(self, quarantine_pipeline_config: Path, tmp_path: Path) -> None:
        """Invalid source rows are written to the quarantine sink.

        Post elspeth-5069612f3c (rows_routed counter split): invalid source rows
        traverse the on_validation_failure DIVERT path, which increments
        rows_routed_failure (NOT rows_routed_success).  The test name reflects
        the new failure-side semantics.

        This is the key acceptance test for the source quarantine routing feature.
        Before this fix, route_to_sink() was a stub and invalid rows were dropped.
        """
        from elspeth.cli import app

        # Run the pipeline
        result = runner.invoke(app, ["run", "-s", str(quarantine_pipeline_config), "--execute"])
        assert result.exit_code == 0

        # Check valid rows went to default output
        output_file = tmp_path / "output.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data) == 2  # alice and carol (valid rows)
        assert {d["name"] for d in data} == {"alice", "carol"}

        # Check invalid row went to quarantine sink
        quarantine_file = tmp_path / "quarantine.json"
        assert quarantine_file.exists(), "Quarantine sink should receive invalid rows"
        quarantine_data = json.loads(quarantine_file.read_text())
        assert len(quarantine_data) == 1  # bob (invalid row)
        assert quarantine_data[0]["name"] == "bob"
        assert quarantine_data[0]["score"] == "bad"  # Original value preserved

    def test_discard_does_not_write_to_sink(self, tmp_path: Path, csv_with_invalid_rows: Path) -> None:
        """When on_validation_failure='discard', invalid rows are not written."""
        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "default",
                    "options": {
                        "path": str(csv_with_invalid_rows),
                        "on_validation_failure": "discard",  # Intentionally drop
                        "schema": {
                            "mode": "fixed",
                            "fields": ["id: int", "name: str", "score: int"],
                        },
                    },
                }
            },
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))

        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])
        assert result.exit_code == 0

        # Only valid rows in output
        output_file = tmp_path / "output.json"
        data = json.loads(output_file.read_text())
        assert len(data) == 2  # alice and carol only


class TestTransformErrorSinkRouting:
    """Tests for sinks only reachable via transform on_error divert edges.

    Verifies that a sink referenced only by on_error doesn't break DAG
    validation. The __error_N__ divert edge makes it reachable in the graph.
    """

    def test_dedicated_error_sink_does_not_break_graph(self, tmp_path: Path) -> None:
        """A sink only referenced by on_error must not break DAG validation."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,value\n1,good\n2,bad\n3,ok\n")

        config = {
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "passthrough_input",
                    "options": {
                        "path": str(input_csv),
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                }
            },
            "transforms": [
                {
                    "name": "passthrough_0",
                    "plugin": "passthrough",
                    "input": "passthrough_input",
                    "on_success": "default",
                    "on_error": "errors",
                    "options": {"schema": {"mode": "observed"}},
                },
            ],
            "sinks": {
                "default": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "output.json"),
                        "schema": {"mode": "observed"},
                    },
                },
                "errors": {
                    "plugin": "json",
                    "on_write_failure": "discard",
                    "options": {
                        "path": str(tmp_path / "errors.json"),
                        "schema": {"mode": "observed"},
                    },
                },
            },
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }

        config_path = tmp_path / "settings.yaml"
        config_path.write_text(yaml.dump(config))

        from elspeth.cli import app

        result = runner.invoke(app, ["run", "-s", str(config_path), "--execute"])
        assert result.exit_code == 0, f"Pipeline failed: {result.output}"


# ===========================================================================
# CLI join command tests (ADR-030 §B.1, slice 5 task c)
# ===========================================================================


def _make_minimal_settings(tmp_path: Path, *, config_hash_override: str | None = None) -> Path:
    """Write a minimal settings.yaml and return the path.

    The config_hash stored in the run row is whatever ``stable_hash(resolve_config(settings))``
    produces for this config.  For refusal tests we either use the real hash or patch the
    run row with a known mismatch value.
    """
    sample_csv = tmp_path / "data.csv"
    sample_csv.write_text("id,name\n1,alice\n")
    config = {
        "sources": {
            "primary": {
                "plugin": "csv",
                "on_success": "default",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            }
        },
        "sinks": {
            "default": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {
                    "path": str(tmp_path / "output.json"),
                    "schema": {"mode": "observed"},
                },
            },
        },
        "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        "payload_store": {"backend": "filesystem", "base_path": str(tmp_path / "payloads")},
    }
    (tmp_path / "payloads").mkdir(exist_ok=True)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


def _seed_running_run(
    tmp_path: Path,
    *,
    run_id: str,
    config_hash: str,
    status: str = "running",
    live_leader: bool = True,
) -> None:
    """Seed the landscape DB with a run row and (optionally) a live coordination seat."""
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.landscape.schema import run_coordination_table, run_workers_table, runs_table

    db_path = tmp_path / "landscape.db"
    db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
    try:
        with db.engine.begin() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id=run_id,
                    started_at=_JOIN_NOW,
                    config_hash=config_hash,
                    settings_json="{}",
                    canonical_version="v1",
                    status=status,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            if live_leader:
                leader_wid = f"worker:{run_id}:leaderabc123"
                expires = (_JOIN_NOW + timedelta(seconds=_JOIN_WINDOW)).replace(tzinfo=None)
                conn.execute(
                    insert(run_coordination_table).values(
                        run_id=run_id,
                        leader_worker_id=leader_wid,
                        leader_epoch=1,
                        leader_heartbeat_expires_at=expires,
                        updated_at=_JOIN_NOW.replace(tzinfo=None),
                    )
                )
                conn.execute(
                    insert(run_workers_table).values(
                        worker_id=leader_wid,
                        run_id=run_id,
                        role="leader",
                        status="active",
                        registered_at=_JOIN_NOW.replace(tzinfo=None),
                        heartbeat_expires_at=expires,
                    )
                )
    finally:
        db.close()


class TestJoinCommand:
    """Tests for ``elspeth join`` CLI (ADR-030 §B.1, slice 5 task c).

    Each test seeds a temporary SQLite DB to the relevant state without going
    through a full pipeline run, keeping the fixture footprint minimal.
    """

    def test_join_requires_settings_or_default(self, tmp_path: Path) -> None:
        """join exits 1 when settings.yaml is absent and no --settings provided."""
        from elspeth.cli import app

        db_path = tmp_path / "landscape.db"

        # Invoke from a temp directory that has NO settings.yaml.
        with runner.isolated_filesystem():
            result = runner.invoke(
                app,
                ["join", "run-missing-settings", "--database", str(db_path)],
            )

        assert result.exit_code == 1
        assert "Settings" in result.output or "settings" in result.output

    def test_join_refused_terminal_run(self, tmp_path: Path) -> None:
        """join exits 1 with a clear refusal when the run is COMPLETED (terminal)."""
        from elspeth.cli import app

        run_id = "run-terminal-001"
        settings_path = _make_minimal_settings(tmp_path)
        _seed_running_run(tmp_path, run_id=run_id, config_hash="any", status="completed", live_leader=False)

        result = runner.invoke(
            app,
            ["join", run_id, "--settings", str(settings_path)],
        )

        assert result.exit_code == 1
        combined = result.output
        assert "terminal" in combined.lower() or "completed" in combined.lower() or "cannot" in combined.lower()

    def test_join_refused_config_mismatch(self, tmp_path: Path) -> None:
        """join exits 1 when the joiner's config hash differs from the run's."""
        from elspeth.cli import app

        run_id = "run-hash-mismatch-002"
        settings_path = _make_minimal_settings(tmp_path)

        # Seed with a deliberately wrong config_hash so the admission check refuses.
        _seed_running_run(tmp_path, run_id=run_id, config_hash="totally-wrong-hash", live_leader=True)

        result = runner.invoke(
            app,
            ["join", run_id, "--settings", str(settings_path)],
        )

        assert result.exit_code == 1
        combined = result.output
        # The refusal message should mention hash or mismatch.
        assert "hash" in combined.lower() or "mismatch" in combined.lower() or "config" in combined.lower()

    def test_join_refused_no_live_leader(self, tmp_path: Path) -> None:
        """join exits 1 and names elspeth resume when the leader seat is dead."""
        from elspeth.cli import app
        from elspeth.core.canonical import stable_hash
        from elspeth.core.config import load_settings, resolve_config

        run_id = "run-dead-leader-003"
        settings_path = _make_minimal_settings(tmp_path)

        # Compute the real config_hash so the hash check passes, then insert an expired seat.
        settings_config = load_settings(settings_path)
        real_hash = stable_hash(resolve_config(settings_config))

        # Seed with live_leader=False: no seat row, so no live leader.
        _seed_running_run(tmp_path, run_id=run_id, config_hash=real_hash, live_leader=False)

        result = runner.invoke(
            app,
            ["join", run_id, "--settings", str(settings_path)],
        )

        assert result.exit_code == 1
        combined = result.output
        assert "leader" in combined.lower() or "resume" in combined.lower()

    def test_join_clean_depart(self, tmp_path: Path) -> None:
        """join exits 0 and emits a departed confirmation after admission + immediate drain completion.

        The drain loop is stubbed to return immediately (simulating zero available
        work + run already terminal) so the test stays deterministic and fast without
        real wall-clock sleeps.
        """
        from unittest.mock import patch

        from elspeth.cli import app
        from elspeth.core.canonical import stable_hash
        from elspeth.core.config import load_settings, resolve_config

        run_id = "run-clean-depart-004"
        settings_path = _make_minimal_settings(tmp_path)

        # Compute the real config_hash so admission succeeds.
        settings_config = load_settings(settings_path)
        real_hash = stable_hash(resolve_config(settings_config))

        # Seed a RUNNING run with a live leader seat whose expiry is well in the
        # future (real datetime.now(UTC) + 3600s) so admit_follower's liveness
        # check passes.
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import run_coordination_table, run_workers_table, runs_table

        real_now = datetime.now(UTC)
        db_path = tmp_path / "landscape.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    insert(runs_table).values(
                        run_id=run_id,
                        started_at=real_now.replace(tzinfo=None),
                        config_hash=real_hash,
                        settings_json="{}",
                        canonical_version="v1",
                        status="running",
                        openrouter_catalog_sha256="0" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )
                leader_wid = f"worker:{run_id}:leaderabc123"
                expires = (real_now + timedelta(hours=1)).replace(tzinfo=None)
                conn.execute(
                    insert(run_coordination_table).values(
                        run_id=run_id,
                        leader_worker_id=leader_wid,
                        leader_epoch=1,
                        leader_heartbeat_expires_at=expires,
                        updated_at=real_now.replace(tzinfo=None),
                    )
                )
                conn.execute(
                    insert(run_workers_table).values(
                        worker_id=leader_wid,
                        run_id=run_id,
                        role="leader",
                        status="active",
                        registered_at=real_now.replace(tzinfo=None),
                        heartbeat_expires_at=expires,
                    )
                )
        finally:
            db.close()

        # Stub build_follower_processor to return a no-op spec'd mock so the
        # drain loop exits immediately.  This tests: admission → worker_id minted
        # → follower built → run() called → clean departure path → exit 0.
        # build_follower_processor would otherwise fail because the seeded run
        # has no registered edges (the full run_core.py pipeline is not invoked).
        from unittest.mock import create_autospec

        from elspeth.engine.orchestrator.follower import FollowerProcessor

        mock_follower = create_autospec(FollowerProcessor, instance=True)
        mock_follower.run.return_value = None  # Immediate clean exit

        with patch(
            "elspeth.engine.orchestrator.follower.build_follower_processor",
            return_value=mock_follower,
        ):
            result = runner.invoke(
                app,
                ["join", run_id, "--settings", str(settings_path)],
            )

        assert result.exit_code == 0, f"Expected exit 0 but got {result.exit_code}.\nOutput:\n{result.output}"
        combined = result.output
        # Should mention the run was joined and departed cleanly.
        assert run_id in combined

    def test_join_startup_failure_departs_admitted_worker_and_cleans_plugins(self, tmp_path: Path) -> None:
        """A follower plugin on_start failure after admission must still depart and tear down."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from sqlalchemy import select

        from elspeth.cli import app
        from elspeth.core.canonical import stable_hash
        from elspeth.core.config import load_settings, resolve_config
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.schema import run_coordination_table, run_workers_table, runs_table

        run_id = "run-startup-failure-departs-005"
        settings_path = _make_minimal_settings(tmp_path)
        settings_config = load_settings(settings_path)
        real_hash = stable_hash(resolve_config(settings_config))

        real_now = datetime.now(UTC)
        db_path = tmp_path / "landscape.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    insert(runs_table).values(
                        run_id=run_id,
                        started_at=real_now.replace(tzinfo=None),
                        config_hash=real_hash,
                        settings_json="{}",
                        canonical_version="v1",
                        status="running",
                        openrouter_catalog_sha256="0" * 64,
                        openrouter_catalog_source="bundled",
                    )
                )
                leader_wid = f"worker:{run_id}:leaderabc123"
                expires = (real_now + timedelta(hours=1)).replace(tzinfo=None)
                conn.execute(
                    insert(run_coordination_table).values(
                        run_id=run_id,
                        leader_worker_id=leader_wid,
                        leader_epoch=1,
                        leader_heartbeat_expires_at=expires,
                        updated_at=real_now.replace(tzinfo=None),
                    )
                )
                conn.execute(
                    insert(run_workers_table).values(
                        worker_id=leader_wid,
                        run_id=run_id,
                        role="leader",
                        status="active",
                        registered_at=real_now.replace(tzinfo=None),
                        heartbeat_expires_at=expires,
                    )
                )
        finally:
            db.close()

        failing_sink = SimpleNamespace(
            name="failing_sink",
            on_start=MagicMock(side_effect=RuntimeError("sink startup failed")),
            on_complete=MagicMock(),
            close=MagicMock(),
        )
        plugins = SimpleNamespace(
            sources={"primary": object()},
            source_settings_map={"primary": object()},
            transforms=[],
            aggregations={},
            sinks={"default": failing_sink},
        )
        execution_graph = MagicMock()
        execution_graph.get_aggregation_id_map.return_value = {}

        with (
            patch("elspeth.plugins.infrastructure.runtime_factory.instantiate_plugins_from_config", return_value=plugins),
            patch("elspeth.cli._build_resume_graphs", return_value=(MagicMock(), execution_graph)),
            patch("elspeth.engine.orchestrator.follower.build_follower_processor") as mock_build_follower,
        ):
            result = runner.invoke(
                app,
                ["join", run_id, "--settings", str(settings_path)],
            )

        assert result.exit_code == 4
        mock_build_follower.assert_called_once()
        failing_sink.on_start.assert_called_once()
        failing_sink.on_complete.assert_called_once()
        failing_sink.close.assert_called_once()

        verify_db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=False)
        try:
            with verify_db.engine.connect() as conn:
                follower_rows = (
                    conn.execute(
                        select(run_workers_table.c.worker_id, run_workers_table.c.status)
                        .where(run_workers_table.c.run_id == run_id)
                        .where(run_workers_table.c.role == "follower")
                    )
                    .mappings()
                    .all()
                )
        finally:
            verify_db.close()

        assert len(follower_rows) == 1
        assert follower_rows[0]["status"] == "departed"

    def test_join_json_format_suppresses_db_banner(self, tmp_path: Path) -> None:
        """join --format json must NOT print the settings-derived DB banner to stdout.

        Regression: the settings-derived database path emitted an unconditional
        ``typer.echo("Using database from settings.yaml: ...")`` to stdout BEFORE the
        JSON events.  A caller parsing stdout as JSON/JSONL would choke on that
        non-JSON line.  The banner is informational and must be suppressed in JSON
        mode, matching the ``resume`` command's ``if output_format != "json"``
        convention.  It still appears in (default) console mode.

        The banner fires during DB resolution — ahead of admission — so a RUNNING
        run with no live leader is sufficient to reach the banner site; admission
        refusing afterwards (exit 1) does not affect the assertion.
        """
        from elspeth.cli import app

        run_id = "run-json-banner-005"
        settings_path = _make_minimal_settings(tmp_path)
        _seed_running_run(tmp_path, run_id=run_id, config_hash="any", live_leader=False)

        banner = "Using database from settings.yaml"

        # Console mode (default): banner expected on stdout.
        console_result = runner.invoke(app, ["join", run_id, "--settings", str(settings_path)])
        assert banner in console_result.output, f"Console mode should print the banner.\nOutput:\n{console_result.output}"

        # JSON mode: banner must NOT pollute stdout.
        json_result = runner.invoke(app, ["join", run_id, "--settings", str(settings_path), "--format", "json"])
        assert banner not in json_result.output, f"JSON mode must suppress the banner.\nOutput:\n{json_result.output}"


# ===========================================================================
# Eviction handling tests — elspeth-c6da3d69f1
# Verify that ``run`` and ``resume`` emit event=evicted / exit 3 (not
# event=error / exit 4) when RunWorkerEvictedError escapes the execution seam.
# ===========================================================================


def _make_jsonl_settings(tmp_path: Path) -> Path:
    """Write a minimal settings.yaml using a JSONL sink (supports resume) and return the path."""
    sample_csv = tmp_path / "data.csv"
    sample_csv.write_text("id,name\n1,alice\n")
    config = {
        "sources": {
            "primary": {
                "plugin": "csv",
                "on_success": "default",
                "options": {
                    "path": str(sample_csv),
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            }
        },
        "sinks": {
            "default": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {
                    "path": str(tmp_path / "output.jsonl"),
                    "schema": {"mode": "observed"},
                },
            },
        },
        "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        "payload_store": {"backend": "filesystem", "base_path": str(tmp_path / "payloads")},
    }
    (tmp_path / "payloads").mkdir(exist_ok=True)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.dump(config))
    return settings_path


class TestRunResumeEviction:
    """Verify that RunWorkerEvictedError is caught by dedicated 'except' arms in
    'run' and 'resume', emitting event=evicted / exit 3 (not event=error / exit 4).

    Regression for elspeth-c6da3d69f1: before the fix these commands fell
    through to 'except Exception' and emitted event=error with a traceback.
    """

    def test_run_evicted_emits_evicted_event(self, tmp_path: Path) -> None:
        """``elspeth run --format json`` exits 3 and emits event=evicted when
        RunWorkerEvictedError escapes _execute_pipeline_with_instances.

        PRE-FIX behaviour: exit 4, event=error (falls through to except Exception).
        POST-FIX behaviour: exit 3, event=evicted (dedicated arm before TIER_1_ERRORS).
        """
        from unittest.mock import patch

        from elspeth.cli import app
        from elspeth.contracts.errors import RunWorkerEvictedError

        settings_path = _make_minimal_settings(tmp_path)

        eviction_exc = RunWorkerEvictedError(worker_id="worker:run-x:abc", run_id="run-x")

        with patch(
            "elspeth.cli._execute_pipeline_with_instances",
            side_effect=eviction_exc,
        ):
            result = runner.invoke(
                app,
                ["run", "--settings", str(settings_path), "--execute", "--format", "json"],
            )

        # Must exit 3 (evicted / interrupted-style), not 4 (unhandled exception).
        assert result.exit_code == 3, f"Expected exit 3 for eviction but got {result.exit_code}.\nOutput:\n{result.output}"

        # Combined stdout+stderr from CliRunner — find the evicted JSON line.
        combined = result.output
        evicted_lines = [line for line in combined.splitlines() if '"event"' in line and '"evicted"' in line]
        assert evicted_lines, f"Expected a JSON line with event=evicted, got none.\nOutput:\n{combined}"

        event = json.loads(evicted_lines[0])
        assert event["event"] == "evicted"
        assert event.get("run_id") == "run-x"
        assert event.get("worker_id") == "worker:run-x:abc"
        assert "message" in event

        # Must NOT emit event=error or a Python traceback.
        assert '"event": "error"' not in combined and '"event":"error"' not in combined, f"Must not emit event=error.\nOutput:\n{combined}"
        assert "Traceback" not in combined, f"Must not emit a traceback.\nOutput:\n{combined}"

    def test_resume_evicted_emits_evicted_event(self, tmp_path: Path) -> None:
        """``elspeth resume --execute --format json`` exits 3 and emits event=evicted
        when RunWorkerEvictedError escapes _execute_resume_with_instances.

        PRE-FIX behaviour: exit 4, event=error (falls through to except Exception).
        POST-FIX behaviour: exit 3, event=evicted (dedicated arm before TIER_1_ERRORS).
        """
        from unittest.mock import MagicMock, patch

        from elspeth.cli import app
        from elspeth.contracts.checkpoint import ResumeCheck
        from elspeth.contracts.errors import RunWorkerEvictedError

        run_id = "run-evict-resume-001"
        settings_path = _make_jsonl_settings(tmp_path)

        # Seed the landscape DB so LandscapeDB.from_url succeeds and the
        # schema-inspection guard passes.
        from elspeth.core.landscape import LandscapeDB

        db_path = tmp_path / "landscape.db"
        seed_db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        seed_db.close()

        eviction_exc = RunWorkerEvictedError(worker_id="worker:resume-x:xyz", run_id=run_id)

        mock_resume_point = MagicMock()
        mock_resume_point.sequence_number = 0
        mock_resume_point.barrier_scalars = None

        with (
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.can_resume",
                return_value=ResumeCheck(can_resume=True),
            ),
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_resume_point",
                return_value=mock_resume_point,
            ),
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_rows",
                return_value=[],
            ),
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.count_blocked_barrier_items",
                return_value=0,
            ),
            patch(
                "elspeth.cli._execute_resume_with_instances",
                side_effect=eviction_exc,
            ),
        ):
            result = runner.invoke(
                app,
                ["resume", run_id, "--settings", str(settings_path), "--execute", "--format", "json"],
            )

        # Must exit 3 (evicted / interrupted-style), not 4 (unhandled exception).
        assert result.exit_code == 3, f"Expected exit 3 for eviction but got {result.exit_code}.\nOutput:\n{result.output}"

        combined = result.output
        evicted_lines = [line for line in combined.splitlines() if '"event"' in line and '"evicted"' in line]
        assert evicted_lines, f"Expected a JSON line with event=evicted, got none.\nOutput:\n{combined}"

        event = json.loads(evicted_lines[0])
        assert event["event"] == "evicted"
        assert event.get("run_id") == run_id
        assert event.get("worker_id") == "worker:resume-x:xyz"
        assert "message" in event

        # Must NOT emit event=error or a Python traceback.
        assert '"event": "error"' not in combined and '"event":"error"' not in combined, f"Must not emit event=error.\nOutput:\n{combined}"
        assert "Traceback" not in combined, f"Must not emit a traceback.\nOutput:\n{combined}"
