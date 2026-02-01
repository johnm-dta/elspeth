# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer
from dynaconf.vendor.ruamel.yaml.parser import ParserError as YamlParserError
from dynaconf.vendor.ruamel.yaml.scanner import ScannerError as YamlScannerError
from pydantic import ValidationError

from elspeth import __version__
from elspeth.contracts import AggregationName, ExecutionResult, ProgressEvent
from elspeth.contracts.events import (
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    RunSummary,
)
from elspeth.core.config import ElspethSettings, load_settings, resolve_config
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.testing.chaosllm.cli import app as chaosllm_app
from elspeth.testing.chaosllm.cli import mcp_app as chaosllm_mcp_app

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import RowPlugin
    from elspeth.plugins.manager import PluginManager
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol

__all__ = [
    "app",
    "load_settings",  # Re-exported from config for convenience
]

# Module-level singleton for plugin manager
_plugin_manager_cache: PluginManager | None = None


def _get_plugin_manager() -> PluginManager:
    """Get initialized plugin manager (singleton).

    Returns:
        PluginManager with all built-in plugins registered
    """
    global _plugin_manager_cache

    from elspeth.plugins.manager import PluginManager

    if _plugin_manager_cache is None:
        manager = PluginManager()
        manager.register_builtin_plugins()
        _plugin_manager_cache = manager
    return _plugin_manager_cache


app = typer.Typer(
    name="elspeth",
    help="ELSPETH: Auditable Sense/Decide/Act pipelines.",
    no_args_is_help=True,
)

app.add_typer(chaosllm_app, name="chaosllm", help="ChaosLLM server commands.")
app.add_typer(chaosllm_mcp_app, name="chaosllm-mcp", help="ChaosLLM MCP analysis tools.")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"elspeth version {__version__}")
        raise typer.Exit()


def _load_dotenv(env_file: Path | None = None) -> bool:
    """Load environment variables from .env file.

    Args:
        env_file: Explicit path to .env file. If None, searches for .env
                 in current directory and parent directories.

    Returns:
        True if .env was found and loaded, False otherwise.

    Raises:
        typer.Exit: If explicit env_file path doesn't exist.
    """
    from dotenv import load_dotenv

    if env_file is not None:
        if not env_file.exists():
            typer.secho(
                f"Error: .env file not found: {env_file}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)
        return load_dotenv(env_file, override=False)

    # load_dotenv searches current dir and parents by default
    return load_dotenv(override=False)  # Don't override existing env vars


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    no_dotenv: bool = typer.Option(
        False,
        "--no-dotenv",
        help="Skip loading .env file.",
    ),
    env_file: Path | None = typer.Option(
        None,
        "--env-file",
        help="Path to .env file (skips automatic search).",
        exists=False,  # We handle existence check ourselves for better error message
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose/debug logging.",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Output structured JSON logs (for machine processing).",
    ),
) -> None:
    """ELSPETH: Auditable Sense/Decide/Act pipelines."""
    # Configure logging at entry point (before any subcommands run)
    # CRITICAL: This must be called early so container logs appear in stdout
    from elspeth.core.logging import configure_logging

    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(json_output=json_logs, level=log_level)

    if not no_dotenv:
        _load_dotenv(env_file=env_file)
    elif env_file is not None:
        typer.secho(
            "Warning: --env-file ignored because --no-dotenv is set.",
            fg=typer.colors.YELLOW,
            err=True,
        )


# === Subcommand stubs (to be implemented in later tasks) ===


@app.command()
def run(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Validate and show what would run without executing.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-x",
        help="Actually execute the pipeline (required for safety).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output.",
    ),
    output_format: Literal["console", "json"] = typer.Option(
        "console",
        "--format",
        "-f",
        help="Output format: 'console' (human-readable) or 'json' (structured JSON).",
    ),
) -> None:
    """Execute a pipeline run.

    Requires --execute flag to actually run (safety feature).
    Use --dry-run to validate configuration without executing.
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config

    settings_path = Path(settings).expanduser()

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except (YamlParserError, YamlScannerError) as e:
        # YAML syntax errors (malformed YAML) - show helpful message
        # e.problem contains the specific error (e.g., "expected ']'", "found a tab")
        typer.echo(f"YAML syntax error in {settings}: {e.problem}", err=True)
        raise typer.Exit(1) from None
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1) from None

    # NEW: Instantiate plugins BEFORE graph construction
    try:
        plugins = instantiate_plugins_from_config(config)
    except Exception as e:
        typer.echo(f"Error instantiating plugins: {e}", err=True)
        raise typer.Exit(1) from None

    # NEW: Build and validate graph from plugin instances
    try:
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except ValueError as e:
        # Schema compatibility errors raised during graph construction (PHASE 2)
        # Updated for Task 4: Schema validation moved to construction time
        typer.echo(f"Schema validation error: {e}", err=True)
        raise typer.Exit(1) from None
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    # Console-only messages (don't emit in JSON mode to keep stream clean)
    if output_format == "console":
        if verbose:
            typer.echo(f"Graph validated: {graph.node_count} nodes, {graph.edge_count} edges")

        if dry_run:
            typer.echo("Dry run mode - would execute:")
            typer.echo(f"  Source: {config.source.plugin}")
            typer.echo(f"  Transforms: {len(config.transforms)}")
            typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
            return

        # Safety check: require explicit --execute flag
        if not execute:
            typer.echo("Pipeline configuration valid.")
            typer.echo(f"  Source: {config.source.plugin}")
            typer.echo("")
            typer.echo("To execute, add --execute (or -x) flag:", err=True)
            typer.echo(f"  elspeth run -s {settings} --execute", err=True)
            raise typer.Exit(1)
    else:
        # JSON mode: early exits without console output
        if dry_run or not execute:
            raise typer.Exit(1)

    # Execute pipeline with pre-instantiated plugins
    try:
        _execute_pipeline_with_instances(
            config,
            graph,
            plugins,
            verbose=verbose,
            output_format=output_format,
        )
    except Exception as e:
        # Emit structured error for JSON mode, human-readable for console
        if output_format == "json":
            import json

            typer.echo(
                json.dumps(
                    {
                        "event": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                ),
                err=True,
            )
        else:
            typer.echo(f"Error during pipeline execution: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def explain(
    run_id: str = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run ID to explain (or 'latest').",
    ),
    row: str | None = typer.Option(
        None,
        "--row",
        help="Row ID to explain.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
    ),
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to Landscape database file (SQLite). Required for explain.",
    ),
    settings: str | None = typer.Option(
        None,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    sink: str | None = typer.Option(
        None,
        "--sink",
        help="Sink name to disambiguate when row has multiple terminal tokens.",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Output text instead of interactive TUI.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Explain lineage for a row or token.

    Use --no-tui for text output or --json for JSON output.
    Without these flags, launches an interactive TUI.

    Examples:

        # JSON output for a specific token
        elspeth explain --run latest --token tok-abc --json --database ./audit.db

        # Text output for a row
        elspeth explain --run run-123 --row row-456 --no-tui --database ./audit.db

        # Interactive TUI
        elspeth explain --run latest --database ./audit.db
    """
    import json as json_module

    from elspeth.cli_helpers import resolve_database_url, resolve_run_id
    from elspeth.core.landscape import (
        LandscapeDB,
        LandscapeRecorder,
        LineageTextFormatter,
        dataclass_to_dict,
    )
    from elspeth.core.landscape import explain as explain_lineage

    if database is None:
        message = "--database is required for explain."
        if json_output:
            typer.echo(json_module.dumps({"error": message}))
        else:
            typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(1) from None

    # Resolve database URL
    settings_path = Path(settings) if settings else None
    try:
        db_url, _ = resolve_database_url(database, settings_path)
    except ValueError as e:
        if json_output:
            typer.echo(json_module.dumps({"error": str(e)}))
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    # Connect to database
    # Initialize db = None for proper cleanup in finally block
    db: LandscapeDB | None = None
    try:
        db = LandscapeDB.from_url(db_url, create_tables=False)
    except Exception as e:
        if json_output:
            typer.echo(json_module.dumps({"error": f"Database connection failed: {e}"}))
        else:
            typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        recorder = LandscapeRecorder(db)

        # Resolve 'latest' run_id
        resolved_run_id = resolve_run_id(run_id, recorder)
        if resolved_run_id is None:
            if json_output:
                typer.echo(json_module.dumps({"error": "No runs found in database"}))
            else:
                typer.echo("Error: No runs found in database", err=True)
            raise typer.Exit(1) from None

        # Must provide either token or row for JSON/no-tui modes
        if (json_output or no_tui) and token is None and row is None:
            if json_output:
                typer.echo(json_module.dumps({"error": "Must provide either --token or --row"}))
            else:
                typer.echo("Error: Must provide either --token or --row", err=True)
            raise typer.Exit(1) from None

        # Query lineage (only for JSON/no-tui modes)
        if json_output or no_tui:
            try:
                lineage_result = explain_lineage(
                    recorder,
                    run_id=resolved_run_id,
                    token_id=token,
                    row_id=row,
                    sink=sink,
                )
            except ValueError as e:
                # Ambiguous row (multiple tokens) or invalid args
                if json_output:
                    typer.echo(json_module.dumps({"error": str(e)}))
                else:
                    typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from None

            if lineage_result is None:
                if json_output:
                    typer.echo(json_module.dumps({"error": "Token or row not found, or no terminal tokens exist yet"}))
                else:
                    typer.echo("Token or row not found, or no terminal tokens exist yet.", err=True)
                raise typer.Exit(1) from None

            # Output based on mode
            if json_output:
                typer.echo(json_module.dumps(dataclass_to_dict(lineage_result), indent=2))
                raise typer.Exit(0)

            if no_tui:
                formatter = LineageTextFormatter()
                typer.echo(formatter.format(lineage_result))
                raise typer.Exit(0)

        # TUI mode
        from elspeth.tui.explain_app import ExplainApp

        tui_app = ExplainApp(
            db=db,
            run_id=resolved_run_id,
            token_id=token,
            row_id=row,
        )
        tui_app.run()

    finally:
        if db is not None:
            db.close()


def _execute_pipeline(
    config: ElspethSettings,
    graph: ExecutionGraph,
    verbose: bool = False,
    output_format: Literal["console", "json"] = "console",
) -> ExecutionResult:
    """Execute a pipeline from configuration.

    NOTE: This function is deprecated in favor of _execute_pipeline_with_instances.
    Telemetry wiring added for P3-2026-02-01 fix.

    Args:
        config: Validated ElspethSettings instance.
        graph: Validated ExecutionGraph instance (must be pre-validated).
        verbose: Show detailed output.
        output_format: Output format ('console' or 'json').

    Returns:
        ExecutionResult with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig

    # Get plugin manager for dynamic plugin lookup
    manager = _get_plugin_manager()

    # Instantiate source via PluginManager
    source_plugin = config.source.plugin
    source_options = dict(config.source.options)

    source_cls = manager.get_source_by_name(source_plugin)
    source: SourceProtocol = source_cls(source_options)

    # Instantiate sinks via PluginManager
    sinks: dict[str, SinkProtocol] = {}
    for sink_name, sink_settings in config.sinks.items():
        sink_plugin = sink_settings.plugin
        sink_options = dict(sink_settings.options)

        sink_cls = manager.get_sink_by_name(sink_plugin)
        sinks[sink_name] = sink_cls(sink_options)

    # Get database URL from settings
    db_url = config.landscape.url
    db = LandscapeDB.from_url(
        db_url,
        dump_to_jsonl=config.landscape.dump_to_jsonl,
        dump_to_jsonl_path=config.landscape.dump_to_jsonl_path,
        dump_to_jsonl_fail_on_error=config.landscape.dump_to_jsonl_fail_on_error,
        dump_to_jsonl_include_payloads=config.landscape.dump_to_jsonl_include_payloads,
        dump_to_jsonl_payload_base_path=(
            str(config.payload_store.base_path)
            if config.landscape.dump_to_jsonl_payload_base_path is None
            else config.landscape.dump_to_jsonl_payload_base_path
        ),
    )

    # Create payload store for audit compliance
    # (CLAUDE.md: "Source entry - Raw data stored before any processing")
    from elspeth.core.payload_store import FilesystemPayloadStore

    if config.payload_store.backend != "filesystem":
        typer.echo(
            f"Error: Unsupported payload store backend '{config.payload_store.backend}'. Only 'filesystem' is currently supported.",
            err=True,
        )
        raise typer.Exit(1)
    payload_store = FilesystemPayloadStore(config.payload_store.base_path)

    # Initialize rate_limit_registry to None so it's defined in finally block
    rate_limit_registry = None

    try:
        # Instantiate transforms from transforms via PluginManager
        transforms: list[RowPlugin] = []
        for plugin_config in config.transforms:
            plugin_name = plugin_config.plugin
            plugin_options = dict(plugin_config.options)

            transform_cls = manager.get_transform_by_name(plugin_name)
            transforms.append(transform_cls(plugin_options))

        # Use the validated graph passed from run() command
        # NOTE: Graph is already validated - do not rebuild it here
        # Build aggregation_settings dict (node_id -> AggregationSettings)
        # Also instantiate aggregation transforms and add to transforms list
        from elspeth.core.config import AggregationSettings

        aggregation_settings: dict[str, AggregationSettings] = {}
        agg_id_map = graph.get_aggregation_id_map()
        for agg_config in config.aggregations:
            node_id = agg_id_map[AggregationName(agg_config.name)]
            aggregation_settings[node_id] = agg_config

            # Instantiate the aggregation transform plugin via PluginManager
            plugin_name = agg_config.plugin
            transform_cls = manager.get_transform_by_name(plugin_name)

            # Merge aggregation options with schema from config
            agg_options = dict(agg_config.options)
            transform = transform_cls(agg_options)

            # Set node_id so processor can identify this as an aggregation node
            transform.node_id = node_id

            # Add to transforms list (after row_plugins transforms)
            transforms.append(transform)

        # Build PipelineConfig with resolved configuration for audit
        pipeline_config = PipelineConfig(
            source=source,
            transforms=transforms,
            sinks=sinks,
            config=resolve_config(config),
            gates=list(config.gates),  # Config-driven gates
            aggregation_settings=aggregation_settings,  # Aggregation configurations
        )

        if verbose:
            typer.echo("Starting pipeline execution...")

        # Create event bus and subscribe progress formatter
        from elspeth.core import EventBus

        event_bus = EventBus()

        # Choose formatters based on output format
        if output_format == "json":
            import json

            # JSON formatters - output structured JSON for each event
            def _format_phase_started_json(event: PhaseStarted) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_started",
                            "phase": event.phase.value,
                            "action": event.action.value,
                            "target": event.target,
                        }
                    )
                )

            def _format_phase_completed_json(event: PhaseCompleted) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_completed",
                            "phase": event.phase.value,
                            "duration_seconds": event.duration_seconds,
                        }
                    )
                )

            def _format_phase_error_json(event: PhaseError) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_error",
                            "phase": event.phase.value,
                            "error": event.error_message,
                            "target": event.target,
                        }
                    ),
                    err=True,
                )

            def _format_run_summary_json(event: RunSummary) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "run_completed",
                            "run_id": event.run_id,
                            "status": event.status.value,
                            "total_rows": event.total_rows,
                            "succeeded": event.succeeded,
                            "failed": event.failed,
                            "quarantined": event.quarantined,
                            "duration_seconds": event.duration_seconds,
                            "exit_code": event.exit_code,
                        }
                    )
                )

            def _format_progress_json(event: ProgressEvent) -> None:
                rate = event.rows_processed / event.elapsed_seconds if event.elapsed_seconds > 0 else 0
                typer.echo(
                    json.dumps(
                        {
                            "event": "progress",
                            "rows_processed": event.rows_processed,
                            "rows_succeeded": event.rows_succeeded,
                            "rows_failed": event.rows_failed,
                            "rows_quarantined": event.rows_quarantined,
                            "elapsed_seconds": event.elapsed_seconds,
                            "rows_per_second": rate,
                        }
                    )
                )

            # Subscribe JSON formatters
            event_bus.subscribe(PhaseStarted, _format_phase_started_json)
            event_bus.subscribe(PhaseCompleted, _format_phase_completed_json)
            event_bus.subscribe(PhaseError, _format_phase_error_json)
            event_bus.subscribe(RunSummary, _format_run_summary_json)
            event_bus.subscribe(ProgressEvent, _format_progress_json)

        else:  # console format (default)
            # Console formatters for human-readable output
            def _format_phase_started(event: PhaseStarted) -> None:
                target_info = f" → {event.target}" if event.target else ""
                typer.echo(f"[{event.phase.value.upper()}] {event.action.value.capitalize()}{target_info}...")

            def _format_phase_completed(event: PhaseCompleted) -> None:
                duration_str = f"{event.duration_seconds:.2f}s" if event.duration_seconds < 60 else f"{event.duration_seconds / 60:.1f}m"
                typer.echo(f"[{event.phase.value.upper()}] ✓ Completed in {duration_str}")

            def _format_phase_error(event: PhaseError) -> None:
                target_info = f" ({event.target})" if event.target else ""
                typer.echo(f"[{event.phase.value.upper()}] ✗ Error{target_info}: {event.error_message}", err=True)

            def _format_run_summary(event: RunSummary) -> None:
                status_symbols = {
                    "completed": "✓",
                    "partial": "⚠",
                    "failed": "✗",
                }
                symbol = status_symbols[event.status.value]
                # Build routed summary with destination breakdown
                routed_summary = ""
                if event.routed > 0:
                    dest_parts = [f"{name}:{count}" for name, count in event.routed_destinations]
                    dest_str = ", ".join(dest_parts) if dest_parts else ""
                    routed_summary = f" | →{event.routed:,} routed"
                    if dest_str:
                        routed_summary += f" ({dest_str})"
                typer.echo(
                    f"\n{symbol} Run {event.status.value.upper()}: "
                    f"{event.total_rows:,} rows processed | "
                    f"✓{event.succeeded:,} succeeded | "
                    f"✗{event.failed:,} failed | "
                    f"⚠{event.quarantined:,} quarantined"
                    f"{routed_summary} | "
                    f"{event.duration_seconds:.2f}s total"
                )

            def _format_progress(event: ProgressEvent) -> None:
                rate = event.rows_processed / event.elapsed_seconds if event.elapsed_seconds > 0 else 0
                typer.echo(
                    f"  Processing: {event.rows_processed:,} rows | "
                    f"{rate:.0f} rows/sec | "
                    f"✓{event.rows_succeeded:,} ✗{event.rows_failed} ⚠{event.rows_quarantined}"
                )

            # Subscribe console formatters
            event_bus.subscribe(PhaseStarted, _format_phase_started)
            event_bus.subscribe(PhaseCompleted, _format_phase_completed)
            event_bus.subscribe(PhaseError, _format_phase_error)
            event_bus.subscribe(RunSummary, _format_run_summary)
            event_bus.subscribe(ProgressEvent, _format_progress)

        # Create runtime configs for external calls and checkpointing
        from elspeth.contracts.config.runtime import (
            RuntimeCheckpointConfig,
            RuntimeConcurrencyConfig,
            RuntimeRateLimitConfig,
            RuntimeTelemetryConfig,
        )
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.rate_limit import RateLimitRegistry
        from elspeth.telemetry import create_telemetry_manager

        rate_limit_config = RuntimeRateLimitConfig.from_settings(config.rate_limit)
        rate_limit_registry = RateLimitRegistry(rate_limit_config)
        concurrency_config = RuntimeConcurrencyConfig.from_settings(config.concurrency)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(config.checkpoint)
        telemetry_config = RuntimeTelemetryConfig.from_settings(config.telemetry)
        telemetry_manager = create_telemetry_manager(telemetry_config)

        # Create checkpoint manager if checkpointing is enabled
        checkpoint_manager = CheckpointManager(db) if checkpoint_config.enabled else None

        # Execute via Orchestrator (creates full audit trail)
        orchestrator = Orchestrator(
            db,
            event_bus=event_bus,
            rate_limit_registry=rate_limit_registry,
            concurrency_config=concurrency_config,
            checkpoint_manager=checkpoint_manager,
            checkpoint_config=checkpoint_config,
            telemetry_manager=telemetry_manager,
        )
        result = orchestrator.run(
            pipeline_config,
            graph=graph,
            settings=config,
            payload_store=payload_store,
        )

        return {
            "run_id": result.run_id,
            "status": result.status,  # RunStatus enum (str subclass)
            "rows_processed": result.rows_processed,
        }
    finally:
        if rate_limit_registry is not None:
            rate_limit_registry.close()
        if telemetry_manager is not None:
            telemetry_manager.close()
        db.close()


def _execute_pipeline_with_instances(
    config: ElspethSettings,
    graph: ExecutionGraph,
    plugins: dict[str, Any],
    verbose: bool = False,
    output_format: Literal["console", "json"] = "console",
) -> ExecutionResult:
    """Execute pipeline using pre-instantiated plugin instances.

    NEW execution path that reuses plugins instantiated during graph construction.
    Eliminates double instantiation.

    Args:
        config: Validated ElspethSettings
        graph: Validated ExecutionGraph (schemas populated)
        plugins: Pre-instantiated plugins from instantiate_plugins_from_config()
        verbose: Show detailed output
        output_format: 'console' or 'json'

    Returns:
        ExecutionResult with run_id, status, rows_processed
    """
    from elspeth.core.config import AggregationSettings
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig

    # Use pre-instantiated plugins with explicit protocol types
    source: SourceProtocol = plugins["source"]
    sinks: dict[str, SinkProtocol] = plugins["sinks"]

    # Build transforms list: row_plugins + aggregations (with node_id)
    transforms: list[RowPlugin] = list(plugins["transforms"])

    # Add aggregation transforms with node_id attached
    agg_id_map = graph.get_aggregation_id_map()
    aggregation_settings: dict[str, AggregationSettings] = {}

    for agg_name, (transform, agg_config) in plugins["aggregations"].items():
        node_id = agg_id_map[agg_name]
        aggregation_settings[node_id] = agg_config

        # Set node_id so processor can identify as aggregation
        transform.node_id = node_id
        transforms.append(transform)

    # Get database
    db_url = config.landscape.url
    db = LandscapeDB.from_url(
        db_url,
        dump_to_jsonl=config.landscape.dump_to_jsonl,
        dump_to_jsonl_path=config.landscape.dump_to_jsonl_path,
        dump_to_jsonl_fail_on_error=config.landscape.dump_to_jsonl_fail_on_error,
        dump_to_jsonl_include_payloads=config.landscape.dump_to_jsonl_include_payloads,
        dump_to_jsonl_payload_base_path=(
            str(config.payload_store.base_path)
            if config.landscape.dump_to_jsonl_payload_base_path is None
            else config.landscape.dump_to_jsonl_payload_base_path
        ),
    )

    # Create payload store for audit compliance
    # (CLAUDE.md: "Source entry - Raw data stored before any processing")
    from elspeth.core.payload_store import FilesystemPayloadStore

    if config.payload_store.backend != "filesystem":
        typer.echo(
            f"Error: Unsupported payload store backend '{config.payload_store.backend}'. Only 'filesystem' is currently supported.",
            err=True,
        )
        raise typer.Exit(1)
    payload_store = FilesystemPayloadStore(config.payload_store.base_path)

    # Initialize rate_limit_registry to None so it's defined in finally block
    rate_limit_registry = None

    try:
        # Build PipelineConfig with pre-instantiated plugins
        pipeline_config = PipelineConfig(
            source=source,
            transforms=transforms,
            sinks=sinks,
            config=resolve_config(config),
            gates=list(config.gates),
            aggregation_settings=aggregation_settings,
        )

        if verbose:
            typer.echo("Starting pipeline execution...")

        # Create event bus and subscribe progress formatter
        from elspeth.core import EventBus

        event_bus = EventBus()

        # Choose formatters based on output format
        if output_format == "json":
            import json

            # JSON formatters - output structured JSON for each event
            def _format_phase_started_json(event: PhaseStarted) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_started",
                            "phase": event.phase.value,
                            "action": event.action.value,
                            "target": event.target,
                        }
                    )
                )

            def _format_phase_completed_json(event: PhaseCompleted) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_completed",
                            "phase": event.phase.value,
                            "duration_seconds": event.duration_seconds,
                        }
                    )
                )

            def _format_phase_error_json(event: PhaseError) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "phase_error",
                            "phase": event.phase.value,
                            "error": event.error_message,
                            "target": event.target,
                        }
                    ),
                    err=True,
                )

            def _format_run_summary_json(event: RunSummary) -> None:
                typer.echo(
                    json.dumps(
                        {
                            "event": "run_completed",
                            "run_id": event.run_id,
                            "status": event.status.value,
                            "total_rows": event.total_rows,
                            "succeeded": event.succeeded,
                            "failed": event.failed,
                            "quarantined": event.quarantined,
                            "duration_seconds": event.duration_seconds,
                            "exit_code": event.exit_code,
                        }
                    )
                )

            def _format_progress_json(event: ProgressEvent) -> None:
                rate = event.rows_processed / event.elapsed_seconds if event.elapsed_seconds > 0 else 0
                typer.echo(
                    json.dumps(
                        {
                            "event": "progress",
                            "rows_processed": event.rows_processed,
                            "rows_succeeded": event.rows_succeeded,
                            "rows_failed": event.rows_failed,
                            "rows_quarantined": event.rows_quarantined,
                            "elapsed_seconds": event.elapsed_seconds,
                            "rows_per_second": rate,
                        }
                    )
                )

            # Subscribe JSON formatters
            event_bus.subscribe(PhaseStarted, _format_phase_started_json)
            event_bus.subscribe(PhaseCompleted, _format_phase_completed_json)
            event_bus.subscribe(PhaseError, _format_phase_error_json)
            event_bus.subscribe(RunSummary, _format_run_summary_json)
            event_bus.subscribe(ProgressEvent, _format_progress_json)

        else:  # console format (default)
            # Console formatters for human-readable output
            def _format_phase_started(event: PhaseStarted) -> None:
                target_info = f" → {event.target}" if event.target else ""
                typer.echo(f"[{event.phase.value.upper()}] {event.action.value.capitalize()}{target_info}...")

            def _format_phase_completed(event: PhaseCompleted) -> None:
                duration_str = f"{event.duration_seconds:.2f}s" if event.duration_seconds < 60 else f"{event.duration_seconds / 60:.1f}m"
                typer.echo(f"[{event.phase.value.upper()}] ✓ Completed in {duration_str}")

            def _format_phase_error(event: PhaseError) -> None:
                target_info = f" ({event.target})" if event.target else ""
                typer.echo(f"[{event.phase.value.upper()}] ✗ Error{target_info}: {event.error_message}", err=True)

            def _format_run_summary(event: RunSummary) -> None:
                status_symbols = {
                    "completed": "✓",
                    "partial": "⚠",
                    "failed": "✗",
                }
                symbol = status_symbols[event.status.value]
                # Build routed summary with destination breakdown
                routed_summary = ""
                if event.routed > 0:
                    dest_parts = [f"{name}:{count}" for name, count in event.routed_destinations]
                    dest_str = ", ".join(dest_parts) if dest_parts else ""
                    routed_summary = f" | →{event.routed:,} routed"
                    if dest_str:
                        routed_summary += f" ({dest_str})"
                typer.echo(
                    f"\n{symbol} Run {event.status.value.upper()}: "
                    f"{event.total_rows:,} rows processed | "
                    f"✓{event.succeeded:,} succeeded | "
                    f"✗{event.failed:,} failed | "
                    f"⚠{event.quarantined:,} quarantined"
                    f"{routed_summary} | "
                    f"{event.duration_seconds:.2f}s total"
                )

            def _format_progress(event: ProgressEvent) -> None:
                rate = event.rows_processed / event.elapsed_seconds if event.elapsed_seconds > 0 else 0
                typer.echo(
                    f"  Processing: {event.rows_processed:,} rows | "
                    f"{rate:.0f} rows/sec | "
                    f"✓{event.rows_succeeded:,} ✗{event.rows_failed} ⚠{event.rows_quarantined}"
                )

            # Subscribe console formatters
            event_bus.subscribe(PhaseStarted, _format_phase_started)
            event_bus.subscribe(PhaseCompleted, _format_phase_completed)
            event_bus.subscribe(PhaseError, _format_phase_error)
            event_bus.subscribe(RunSummary, _format_run_summary)
            event_bus.subscribe(ProgressEvent, _format_progress)

        # Create runtime configs for external calls and checkpointing
        from elspeth.contracts.config.runtime import (
            RuntimeCheckpointConfig,
            RuntimeConcurrencyConfig,
            RuntimeRateLimitConfig,
            RuntimeTelemetryConfig,
        )
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.rate_limit import RateLimitRegistry
        from elspeth.telemetry import create_telemetry_manager

        rate_limit_config = RuntimeRateLimitConfig.from_settings(config.rate_limit)
        rate_limit_registry = RateLimitRegistry(rate_limit_config)
        concurrency_config = RuntimeConcurrencyConfig.from_settings(config.concurrency)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(config.checkpoint)
        telemetry_config = RuntimeTelemetryConfig.from_settings(config.telemetry)
        telemetry_manager = create_telemetry_manager(telemetry_config)

        # Create checkpoint manager if checkpointing is enabled
        checkpoint_manager = CheckpointManager(db) if checkpoint_config.enabled else None

        # Execute via Orchestrator (creates full audit trail)
        orchestrator = Orchestrator(
            db,
            event_bus=event_bus,
            rate_limit_registry=rate_limit_registry,
            concurrency_config=concurrency_config,
            checkpoint_manager=checkpoint_manager,
            checkpoint_config=checkpoint_config,
            telemetry_manager=telemetry_manager,
        )
        result = orchestrator.run(
            pipeline_config,
            graph=graph,
            settings=config,
            payload_store=payload_store,
        )

        return {
            "run_id": result.run_id,
            "status": result.status,  # RunStatus enum (str subclass)
            "rows_processed": result.rows_processed,
        }
    finally:
        if rate_limit_registry is not None:
            rate_limit_registry.close()
        if telemetry_manager is not None:
            telemetry_manager.close()
        db.close()


def _format_validation_error(
    title: str,
    message: str,
    hint: str | None = None,
    details: list[str] | None = None,
) -> None:
    """Display a formatted validation error with optional hint and details."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(stderr=True)

    # Build error content
    content = Text()
    content.append(message, style="white")

    if details:
        content.append("\n\n")
        for detail in details:
            content.append(f"  • {detail}\n", style="dim")

    if hint:
        content.append("\n")
        content.append("Hint: ", style="yellow bold")
        content.append(hint, style="yellow")

    panel = Panel(
        content,
        title=f"[red bold]❌ {title}[/]",
        border_style="red",
        padding=(0, 1),
    )
    console.print(panel)


@app.command()
def validate(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
) -> None:
    """Validate pipeline configuration without running."""
    from elspeth.cli_helpers import instantiate_plugins_from_config

    settings_path = Path(settings).expanduser()

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except (YamlParserError, YamlScannerError) as e:
        # YAML syntax errors (malformed YAML) - show helpful message
        _format_validation_error(
            title="YAML Syntax Error",
            message=f"Failed to parse {settings_path.name}",
            details=[str(e.problem)] if hasattr(e, "problem") else None,
            hint="Check for unclosed brackets, incorrect indentation, or invalid characters.",
        )
        raise typer.Exit(1) from None
    except FileNotFoundError:
        _format_validation_error(
            title="File Not Found",
            message=f"Settings file does not exist: {settings}",
            hint="Check the path and ensure the file exists.",
        )
        raise typer.Exit(1) from None
    except ValidationError as e:
        # Pydantic validation errors (must be before ValueError - ValidationError inherits from it!)
        details = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            details.append(f"{loc}: {error['msg']}")
        _format_validation_error(
            title="Configuration Validation Failed",
            message=f"Invalid settings in {settings_path.name}",
            details=details,
            hint="Check field names, types, and required values.",
        )
        raise typer.Exit(1) from None
    except ValueError as e:
        # Environment variable expansion errors (must be AFTER ValidationError!)
        error_msg = str(e)
        if "environment variable" in error_msg.lower():
            # Extract variable name from error message if possible
            import re

            match = re.search(r"'(\w+)'", error_msg)
            var_name = match.group(1) if match else "VARIABLE"
            _format_validation_error(
                title="Missing Environment Variable",
                message=error_msg,
                hint=f'Set the variable: export {var_name}="your-value"\n         Or use optional syntax: ${{{var_name}:-default}}',
            )
        else:
            _format_validation_error(
                title="Configuration Error",
                message=error_msg,
            )
        raise typer.Exit(1) from None

    # Instantiate plugins BEFORE graph construction
    try:
        plugins = instantiate_plugins_from_config(config)
    except ValueError as e:
        # Plugin configuration errors (e.g., invalid schema for sink type)
        error_msg = str(e)
        _format_validation_error(
            title="Plugin Configuration Error",
            message=error_msg,
            hint="Check plugin options match the plugin's requirements.",
        )
        raise typer.Exit(1) from None
    except Exception as e:
        _format_validation_error(
            title="Plugin Instantiation Failed",
            message=str(e),
            hint="Check that all plugin options are valid and dependencies are available.",
        )
        raise typer.Exit(1) from None

    # Build and validate graph from plugin instances
    try:
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            default_sink=config.default_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except ValueError as e:
        # Schema compatibility errors raised during graph construction
        _format_validation_error(
            title="Schema Validation Error",
            message=str(e),
            hint="Ensure upstream nodes provide fields required by downstream nodes.",
        )
        raise typer.Exit(1) from None
    except GraphValidationError as e:
        _format_validation_error(
            title="Pipeline Graph Error",
            message=str(e),
            hint="Check for cycles, missing sinks, or invalid routing.",
        )
        raise typer.Exit(1) from None

    typer.echo("✅ Pipeline configuration valid!")
    typer.echo(f"  Source: {config.source.plugin}")
    typer.echo(f"  Transforms: {len(config.transforms)}")
    typer.echo(f"  Aggregations: {len(config.aggregations)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")


# Plugins subcommand group
plugins_app = typer.Typer(help="Plugin management commands.")
app.add_typer(plugins_app, name="plugins")


@dataclass(frozen=True)
class PluginInfo:
    """Metadata for a registered plugin.

    Attributes:
        name: The plugin identifier used in configuration files.
        description: Human-readable description of the plugin's purpose.
    """

    name: str
    description: str


def _build_plugin_registry() -> dict[str, list[PluginInfo]]:
    """Build plugin registry dynamically from discovered plugins.

    Uses PluginManager to discover all plugins and extracts descriptions
    from their docstrings.

    Returns:
        Dict mapping plugin type to list of PluginInfo for each plugin.
    """
    from elspeth.plugins.discovery import get_plugin_description

    manager = _get_plugin_manager()

    return {
        "source": [PluginInfo(name=cls.name, description=get_plugin_description(cls)) for cls in manager.get_sources()],
        "transform": [PluginInfo(name=cls.name, description=get_plugin_description(cls)) for cls in manager.get_transforms()],
        "sink": [PluginInfo(name=cls.name, description=get_plugin_description(cls)) for cls in manager.get_sinks()],
    }


@plugins_app.command("list")
def plugins_list(
    plugin_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by plugin type (source, transform, sink).",
    ),
) -> None:
    """List available plugins."""
    # Build registry dynamically from discovered plugins
    registry = _build_plugin_registry()
    valid_types = set(registry.keys())

    if plugin_type and plugin_type not in valid_types:
        typer.echo(f"Error: Invalid type '{plugin_type}'.", err=True)
        typer.echo(f"Valid types: {', '.join(sorted(valid_types))}", err=True)
        raise typer.Exit(1)

    types_to_show = [plugin_type] if plugin_type else list(registry.keys())

    for ptype in types_to_show:
        # types_to_show only contains keys from registry (either filtered by validated plugin_type
        # or directly from registry.keys()), so direct access is safe
        plugins = registry[ptype]
        if plugins:
            typer.echo(f"\n{ptype.upper()}S:")
            for plugin in plugins:
                typer.echo(f"  {plugin.name:20} - {plugin.description}")
        else:
            typer.echo(f"\n{ptype.upper()}S:")
            typer.echo("  (none available)")

    typer.echo()  # Final newline


@app.command()
def purge(
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to Landscape database file (SQLite).",
    ),
    payload_dir: str | None = typer.Option(
        None,
        "--payload-dir",
        "-p",
        help="Path to payload storage directory.",
    ),
    retention_days: int | None = typer.Option(
        None,
        "--retention-days",
        "-r",
        help="Delete payloads older than this many days (default: from config or 90).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without deleting.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Purge old payloads to free storage.

    Deletes PayloadStore blobs older than retention period.
    Landscape metadata (hashes) is preserved for audit trail.

    Examples:

        # See what would be deleted
        elspeth purge --dry-run --database ./landscape.db

        # Delete payloads older than 30 days
        elspeth purge --retention-days 30 --yes --database ./landscape.db
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.core.retention.purge import PurgeManager

    # Try to load settings from settings.yaml if database not provided
    db_url: str | None = None
    payload_path: Path | None = None
    effective_retention_days: int = 90  # Fallback default
    config: ElspethSettings | None = None

    # Try loading settings.yaml first (for payload_store config)
    settings_path = Path("settings.yaml")
    if settings_path.exists():
        try:
            config = load_settings(settings_path)
        except Exception as e:
            if not database:
                # Only fail if we needed settings for database URL
                typer.echo(f"Error loading settings.yaml: {e}", err=True)
                typer.echo("Specify --database to provide path directly.", err=True)
                raise typer.Exit(1) from None
            # Otherwise warn but continue with CLI-provided database
            typer.echo(f"Warning: Could not load settings.yaml: {e}", err=True)

    if database:
        db_path = Path(database).expanduser().resolve()
        # Fail fast with clear error if file doesn't exist
        # (prevents silent creation of empty DB on typoed paths)
        if not db_path.exists():
            typer.echo(f"Error: Database file not found: {db_path}", err=True)
            raise typer.Exit(1) from None
        db_url = f"sqlite:///{db_path}"
    elif config:
        db_url = config.landscape.url
        typer.echo(f"Using database from settings.yaml: {db_url}")
    else:
        typer.echo("Error: No settings.yaml found and --database not provided.", err=True)
        typer.echo("Specify --database to provide path to Landscape database.", err=True)
        raise typer.Exit(1) from None

    # Determine payload path: CLI override > config > default
    if payload_dir:
        payload_path = Path(payload_dir).expanduser()
    elif config:
        # Use config's payload_store.base_path
        payload_path = config.payload_store.base_path.expanduser()
        # Verify backend is supported
        if config.payload_store.backend != "filesystem":
            typer.echo(
                f"Error: Payload store backend '{config.payload_store.backend}' is not supported for purge. "
                f"Only 'filesystem' backend is currently implemented.",
                err=True,
            )
            raise typer.Exit(1) from None
        typer.echo(f"Using payload directory from config: {payload_path}")
    else:
        # Default to ./payloads relative to database location
        if database:
            payload_path = Path(database).parent / "payloads"
        else:
            payload_path = Path("payloads")

    # Determine retention days: CLI override > config > default (90)
    if retention_days is not None:
        effective_retention_days = retention_days
    elif config:
        effective_retention_days = config.payload_store.retention_days
        typer.echo(f"Using retention_days from config: {effective_retention_days}")
    # else: use the fallback default of 90

    # Initialize database and payload store
    # Note: purge is read-only for audit data, no JSONL journaling needed
    try:
        db = LandscapeDB.from_url(db_url)
    except Exception as e:
        typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        payload_store = FilesystemPayloadStore(payload_path)
        purge_manager = PurgeManager(db, payload_store)

        # Find all expired payload refs (rows, calls, routing reasons)
        expired_refs = purge_manager.find_expired_payload_refs(effective_retention_days)

        if not expired_refs:
            typer.echo(f"No payloads older than {effective_retention_days} days found.")
            return

        if dry_run:
            typer.echo(f"Would delete {len(expired_refs)} payload(s) older than {effective_retention_days} days:")
            for ref in expired_refs[:10]:  # Show first 10
                exists = payload_store.exists(ref)
                status = "exists" if exists else "already deleted"
                typer.echo(f"  {ref[:16]}... ({status})")
            if len(expired_refs) > 10:
                typer.echo(f"  ... and {len(expired_refs) - 10} more")
            return

        # Confirm unless --yes
        if not yes:
            confirm = typer.confirm(f"Delete {len(expired_refs)} payload(s) older than {effective_retention_days} days?")
            if not confirm:
                typer.echo("Aborted.")
                raise typer.Exit(1)

        # Execute purge
        result = purge_manager.purge_payloads(expired_refs)

        typer.echo(f"Purge completed in {result.duration_seconds:.2f}s:")
        typer.echo(f"  Deleted: {result.deleted_count}")
        typer.echo(f"  Skipped (not found): {result.skipped_count}")
        if result.failed_refs:
            typer.echo(f"  Failed: {len(result.failed_refs)}")
            for ref in result.failed_refs[:5]:
                typer.echo(f"    {ref[:16]}...")
    finally:
        db.close()


def _execute_resume_with_instances(
    config: ElspethSettings,
    graph: ExecutionGraph,
    plugins: dict[str, Any],
    resume_point: Any,
    payload_store: PayloadStore | None,
    db: LandscapeDB,
) -> Any:  # Returns RunResult from orchestrator.resume()
    """Execute resume using pre-instantiated plugins.

    Similar to _execute_pipeline_with_instances but for resume operations.

    Args:
        config: Validated ElspethSettings
        graph: Validated ExecutionGraph
        plugins: Pre-instantiated plugins (with NullSource)
        resume_point: Resume point information
        payload_store: Payload store for retrieving row data
        db: LandscapeDB connection

    Returns:
        RunResult from orchestrator.resume()
    """
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import AggregationSettings
    from elspeth.engine import Orchestrator, PipelineConfig

    # Use pre-instantiated plugins with explicit protocol types
    source: SourceProtocol = plugins["source"]
    sinks: dict[str, SinkProtocol] = plugins["sinks"]

    # Build transforms list: row_plugins + aggregations (with node_id)
    transforms: list[RowPlugin] = list(plugins["transforms"])

    # Add aggregation transforms with node_id attached
    agg_id_map = graph.get_aggregation_id_map()
    aggregation_settings: dict[str, AggregationSettings] = {}

    for agg_name, (transform, agg_config) in plugins["aggregations"].items():
        node_id = agg_id_map[agg_name]
        aggregation_settings[node_id] = agg_config

        # Set node_id so processor can identify as aggregation
        transform.node_id = node_id
        transforms.append(transform)

    # Build PipelineConfig with pre-instantiated plugins
    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,
        sinks=sinks,
        config=resolve_config(config),
        gates=list(config.gates),
        aggregation_settings=aggregation_settings,
    )

    # Create event bus for progress reporting
    from elspeth.contracts.events import (
        PhaseCompleted,
        PhaseError,
        PhaseStarted,
        RunSummary,
    )
    from elspeth.core import EventBus

    event_bus = EventBus()

    # Console formatters for human-readable output
    def _format_phase_started(event: PhaseStarted) -> None:
        target_info = f" → {event.target}" if event.target else ""
        typer.echo(f"[{event.phase.value.upper()}] {event.action.value.capitalize()}{target_info}...")

    def _format_phase_completed(event: PhaseCompleted) -> None:
        duration_str = f"{event.duration_seconds:.2f}s" if event.duration_seconds < 60 else f"{event.duration_seconds / 60:.1f}m"
        typer.echo(f"[{event.phase.value.upper()}] ✓ Completed in {duration_str}")

    def _format_phase_error(event: PhaseError) -> None:
        target_info = f" ({event.target})" if event.target else ""
        typer.echo(f"[{event.phase.value.upper()}] ✗ Error{target_info}: {event.error_message}", err=True)

    def _format_run_summary(event: RunSummary) -> None:
        status_symbols = {
            "completed": "✓",
            "partial": "⚠",
            "failed": "✗",
        }
        symbol = status_symbols[event.status.value]
        # Build routed summary with destination breakdown
        routed_summary = ""
        if event.routed > 0:
            dest_parts = [f"{name}:{count}" for name, count in event.routed_destinations]
            dest_str = ", ".join(dest_parts) if dest_parts else ""
            routed_summary = f" | →{event.routed:,} routed"
            if dest_str:
                routed_summary += f" ({dest_str})"
        typer.echo(
            f"\n{symbol} Resume {event.status.value.upper()}: "
            f"{event.total_rows:,} rows processed | "
            f"✓{event.succeeded:,} succeeded | "
            f"✗{event.failed:,} failed | "
            f"⚠{event.quarantined:,} quarantined"
            f"{routed_summary} | "
            f"{event.duration_seconds:.2f}s total"
        )

    def _format_progress(event: ProgressEvent) -> None:
        rate = event.rows_processed / event.elapsed_seconds if event.elapsed_seconds > 0 else 0
        typer.echo(
            f"  Processing: {event.rows_processed:,} rows | "
            f"{rate:.0f} rows/sec | "
            f"✓{event.rows_succeeded:,} ✗{event.rows_failed} ⚠{event.rows_quarantined}"
        )

    # Subscribe console formatters
    event_bus.subscribe(PhaseStarted, _format_phase_started)
    event_bus.subscribe(PhaseCompleted, _format_phase_completed)
    event_bus.subscribe(PhaseError, _format_phase_error)
    event_bus.subscribe(RunSummary, _format_run_summary)
    event_bus.subscribe(ProgressEvent, _format_progress)

    # Create runtime configs for external calls and checkpointing
    from elspeth.contracts.config.runtime import (
        RuntimeCheckpointConfig,
        RuntimeConcurrencyConfig,
        RuntimeRateLimitConfig,
        RuntimeTelemetryConfig,
    )
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.telemetry import create_telemetry_manager

    # Initialize to None so they're defined in finally block even if creation fails
    rate_limit_registry = None
    telemetry_manager = None

    try:
        rate_limit_config = RuntimeRateLimitConfig.from_settings(config.rate_limit)
        rate_limit_registry = RateLimitRegistry(rate_limit_config)
        concurrency_config = RuntimeConcurrencyConfig.from_settings(config.concurrency)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(config.checkpoint)
        telemetry_config = RuntimeTelemetryConfig.from_settings(config.telemetry)
        telemetry_manager = create_telemetry_manager(telemetry_config)

        # Create checkpoint manager and orchestrator for resume
        checkpoint_manager = CheckpointManager(db)
        orchestrator = Orchestrator(
            db,
            event_bus=event_bus,
            checkpoint_manager=checkpoint_manager,
            checkpoint_config=checkpoint_config,
            rate_limit_registry=rate_limit_registry,
            concurrency_config=concurrency_config,
            telemetry_manager=telemetry_manager,
        )

        # Execute resume (payload_store is required for resume)
        if payload_store is None:
            raise ValueError("payload_store is required for resume operations")
        result = orchestrator.resume(
            resume_point=resume_point,
            config=pipeline_config,
            graph=graph,
            payload_store=payload_store,
            settings=config,
        )

        return result
    finally:
        # Clean up rate limit registry and telemetry (always, even on failure)
        if rate_limit_registry is not None:
            rate_limit_registry.close()
        if telemetry_manager is not None:
            telemetry_manager.close()


def _build_validation_graph(settings_config: ElspethSettings) -> ExecutionGraph:
    """Build execution graph for resume topology validation.

    CRITICAL: Uses the ORIGINAL source plugin configuration (not NullSource)
    to match the topology hash computed during the original run.

    The checkpoint's upstream_topology_hash was computed with the real source,
    so validation must use the same source to avoid false topology mismatches.

    Returns:
        ExecutionGraph with original source for topology validation
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config

    plugins = instantiate_plugins_from_config(settings_config)

    graph = ExecutionGraph.from_plugin_instances(
        source=plugins["source"],  # Use ORIGINAL source, not NullSource
        transforms=plugins["transforms"],
        sinks=plugins["sinks"],
        aggregations=plugins["aggregations"],
        gates=list(settings_config.gates),
        default_sink=settings_config.default_sink,
        coalesce_settings=list(settings_config.coalesce) if settings_config.coalesce else None,
    )

    graph.validate()
    return graph


def _build_execution_graph(settings_config: ElspethSettings) -> ExecutionGraph:
    """Build execution graph for resume execution.

    Uses NullSource because resume data comes from stored payloads,
    not from re-reading the original source.

    Returns:
        ExecutionGraph with NullSource for execution
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.plugins.sources.null_source import NullSource

    plugins = instantiate_plugins_from_config(settings_config)

    # Override source with NullSource for resume execution
    null_source = NullSource({})
    resume_plugins = {**plugins, "source": null_source}

    graph = ExecutionGraph.from_plugin_instances(
        source=resume_plugins["source"],
        transforms=resume_plugins["transforms"],
        sinks=resume_plugins["sinks"],
        aggregations=resume_plugins["aggregations"],
        gates=list(settings_config.gates),
        default_sink=settings_config.default_sink,
        coalesce_settings=list(settings_config.coalesce) if settings_config.coalesce else None,
    )

    graph.validate()
    return graph


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to Landscape database file (SQLite).",
    ),
    settings_file: str | None = typer.Option(
        None,
        "--settings",
        "-s",
        help="Path to settings YAML file (default: settings.yaml).",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-x",
        help="Actually execute the resume (default is dry-run).",
    ),
) -> None:
    """Resume a failed run from its last checkpoint.

    By default, shows what WOULD happen (dry run). Use --execute to
    actually resume processing.

    Examples:

        # Dry run - show resume info
        elspeth resume run-abc123

        # Actually resume processing
        elspeth resume run-abc123 --execute

        # Resume with explicit database path
        elspeth resume run-abc123 --database ./landscape.db --execute
    """
    from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
    from elspeth.core.landscape import LandscapeDB

    # Settings are REQUIRED for topology validation
    settings_config: ElspethSettings | None = None
    settings_path = Path(settings_file).expanduser() if settings_file else Path("settings.yaml")

    if not settings_path.exists():
        typer.echo(f"Error: Settings file not found: {settings_path}", err=True)
        typer.echo("Settings are required to validate checkpoint compatibility.", err=True)
        raise typer.Exit(1)

    try:
        settings_config = load_settings(settings_path)
    except Exception as e:
        typer.echo(f"Error loading {settings_path}: {e}", err=True)
        raise typer.Exit(1) from None

    # Resolve database URL
    if database:
        db_path = Path(database).expanduser().resolve()
        # Fail fast with clear error if file doesn't exist
        # (prevents silent creation of empty DB on typoed paths)
        if not db_path.exists():
            typer.echo(f"Error: Database file not found: {db_path}", err=True)
            raise typer.Exit(1) from None
        db_url = f"sqlite:///{db_path}"
    else:
        db_url = settings_config.landscape.url
        typer.echo(f"Using database from settings.yaml: {db_url}")

    # Initialize database and recovery manager
    try:
        db = LandscapeDB.from_url(db_url)
    except Exception as e:
        typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        checkpoint_manager = CheckpointManager(db)
        recovery_manager = RecoveryManager(db, checkpoint_manager)

        # Build graph for topology validation (uses original source)
        try:
            validation_graph = _build_validation_graph(settings_config)
        except Exception as e:
            typer.echo(f"Error building validation graph: {e}", err=True)
            raise typer.Exit(1) from None

        # Check if run can be resumed (with topology validation)
        check = recovery_manager.can_resume(run_id, validation_graph)

        if not check.can_resume:
            typer.echo(f"Cannot resume run {run_id}: {check.reason}", err=True)
            raise typer.Exit(1)

        # Get resume point information
        resume_point = recovery_manager.get_resume_point(run_id, validation_graph)
        if resume_point is None:
            typer.echo(f"Error: Could not get resume point for run {run_id}", err=True)
            raise typer.Exit(1)

        # Get count of unprocessed rows
        unprocessed_row_ids = recovery_manager.get_unprocessed_rows(run_id)

        # Display resume point information
        typer.echo(f"Run {run_id} can be resumed.")
        typer.echo("\nResume point:")
        typer.echo(f"  Token ID: {resume_point.token_id}")
        typer.echo(f"  Node ID: {resume_point.node_id}")
        typer.echo(f"  Sequence number: {resume_point.sequence_number}")
        if resume_point.aggregation_state:
            typer.echo("  Has aggregation state: Yes")
        else:
            typer.echo("  Has aggregation state: No")
        typer.echo(f"  Unprocessed rows: {len(unprocessed_row_ids)}")

        if not execute:
            typer.echo("\nDry run - use --execute to actually resume processing.")
            typer.echo("Topology validation passed - checkpoint is compatible with current config.")
            return

        # Execute resume (graph already built above for validation)
        typer.echo(f"\nResuming run {run_id}...")

        # Get payload store from settings
        from elspeth.core.payload_store import FilesystemPayloadStore

        if settings_config.payload_store.backend != "filesystem":
            typer.echo(
                f"Error: Unsupported payload store backend '{settings_config.payload_store.backend}'. "
                f"Only 'filesystem' is currently supported.",
                err=True,
            )
            raise typer.Exit(1)

        payload_path = settings_config.payload_store.base_path
        if not payload_path.exists():
            typer.echo(f"Error: Payload directory not found: {payload_path}", err=True)
            raise typer.Exit(1)

        payload_store = FilesystemPayloadStore(payload_path)

        # Build execution graph (uses NullSource for resume)
        try:
            execution_graph = _build_execution_graph(settings_config)
        except Exception as e:
            typer.echo(f"Error building execution graph: {e}", err=True)
            raise typer.Exit(1) from None

        # Instantiate plugins for execution
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.plugins.sources.null_source import NullSource

        try:
            plugins = instantiate_plugins_from_config(settings_config)
        except Exception as e:
            typer.echo(f"Error instantiating plugins: {e}", err=True)
            raise typer.Exit(1) from None

        # CRITICAL: Validate and configure sinks for resume mode
        # Each sink declares whether it supports resume and self-configures
        manager = _get_plugin_manager()
        resume_sinks = {}

        for sink_name, sink_config in settings_config.sinks.items():
            sink_cls = manager.get_sink_by_name(sink_config.plugin)
            sink_options = dict(sink_config.options)

            # Instantiate sink to check resume capability
            try:
                sink = sink_cls(sink_options)
            except Exception as e:
                typer.echo(f"Error creating sink '{sink_name}': {e}", err=True)
                raise typer.Exit(1) from None

            # Check if sink supports resume
            if not sink.supports_resume:
                typer.echo(
                    f"Error: Cannot resume with sink '{sink_name}' (plugin: {sink_config.plugin}). "
                    f"This sink does not support resume/append mode.\n"
                    f"Hint: Use a different sink type or start a new run.",
                    err=True,
                )
                raise typer.Exit(1)

            # Configure sink for resume (switches to append mode)
            try:
                sink.configure_for_resume()
            except NotImplementedError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(1) from None

            # Validate output target schema compatibility
            validation = sink.validate_output_target()
            if not validation.valid:
                typer.echo(
                    f"\nError: Cannot resume with sink '{sink_name}'.\nOutput target schema mismatch: {validation.error_message}",
                    err=True,
                )
                if validation.missing_fields:
                    typer.echo(f"  Missing fields: {list(validation.missing_fields)}", err=True)
                if validation.extra_fields:
                    typer.echo(f"  Extra fields: {list(validation.extra_fields)}", err=True)
                if validation.order_mismatch:
                    typer.echo(
                        "  Note: Fields present but in wrong order (strict mode)",
                        err=True,
                    )
                typer.echo(
                    "\nHint: Either fix the output file to match schema, or start a new run.",
                    err=True,
                )
                raise typer.Exit(1)

            resume_sinks[sink_name] = sink

        # Override source with NullSource for resume (data comes from payloads)
        null_source = NullSource({})
        resume_plugins = {
            **plugins,
            "source": null_source,
            "sinks": resume_sinks,  # Use append-mode sinks
        }

        # Execute resume with execution graph (NullSource)
        try:
            result = _execute_resume_with_instances(
                config=settings_config,
                graph=execution_graph,
                plugins=resume_plugins,
                resume_point=resume_point,
                payload_store=payload_store,
                db=db,
            )
        except Exception as e:
            typer.echo(f"Error during resume: {e}", err=True)
            raise typer.Exit(1) from None

        typer.echo("\nResume complete:")
        typer.echo(f"  Rows processed: {result.rows_processed}")
        typer.echo(f"  Rows succeeded: {result.rows_succeeded}")
        typer.echo(f"  Rows failed: {result.rows_failed}")
        typer.echo(f"  Status: {result.status.value}")

    finally:
        db.close()


@app.command()
def health(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Include detailed check information.",
    ),
) -> None:
    """Check system health for deployment verification.

    Verifies that ELSPETH is properly configured and can connect to
    required services. Used by deployment scripts and container health checks.

    Examples:

        # Basic health check
        elspeth health

        # JSON output for automation
        elspeth health --json

        # Verbose with details
        elspeth health --verbose
    """
    import json as json_module
    import os
    import subprocess
    import sys

    from elspeth import __version__

    # Health check results
    checks: dict[str, dict[str, str | bool]] = {}
    overall_healthy = True

    # Check 1: Version (always passes if we got here)
    checks["version"] = {
        "status": "ok",
        "value": __version__,
    }

    # Check 2: Git commit SHA (if available)
    git_sha = os.environ.get("GIT_COMMIT_SHA", "")
    if not git_sha:
        # Try to get from git
        try:
            git_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if git_result.returncode == 0:
                git_sha = git_result.stdout.strip()
        except Exception:
            git_sha = "unknown"

    checks["commit"] = {
        "status": "ok" if git_sha and git_sha != "unknown" else "warn",
        "value": git_sha or "unknown",
    }

    # Check 3: Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks["python"] = {
        "status": "ok",
        "value": python_version,
    }

    # Check 4: Database connectivity (if DATABASE_URL is set)
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = {
                "status": "ok",
                "value": "connected",
            }
        except Exception as e:
            checks["database"] = {
                "status": "error",
                "value": str(e),
            }
            overall_healthy = False
    else:
        checks["database"] = {
            "status": "skip",
            "value": "DATABASE_URL not set",
        }

    # Check 5: Config directory readable (container path)
    config_paths = ["/app/config", "./config"]
    config_readable = False
    config_path_used = ""
    for path in config_paths:
        if os.path.isdir(path) and os.access(path, os.R_OK):
            config_readable = True
            config_path_used = path
            break

    if config_readable:
        checks["config_dir"] = {
            "status": "ok",
            "value": config_path_used,
        }
    else:
        checks["config_dir"] = {
            "status": "warn",
            "value": "not found or not readable",
        }

    # Check 6: Output directory writable (container path)
    output_paths = ["/app/output", "./output"]
    output_writable = False
    output_path_used = ""
    for path in output_paths:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            output_writable = True
            output_path_used = path
            break

    if output_writable:
        checks["output_dir"] = {
            "status": "ok",
            "value": output_path_used,
        }
    else:
        checks["output_dir"] = {
            "status": "warn",
            "value": "not found or not writable",
        }

    # Check 7: Plugins loaded
    try:
        manager = _get_plugin_manager()
        source_count = len(manager.get_sources())
        transform_count = len(manager.get_transforms())
        sink_count = len(manager.get_sinks())
        checks["plugins"] = {
            "status": "ok",
            "value": f"{source_count} sources, {transform_count} transforms, {sink_count} sinks",
        }
    except Exception as e:
        checks["plugins"] = {
            "status": "error",
            "value": str(e),
        }
        overall_healthy = False

    # Determine overall status
    status = "healthy" if overall_healthy else "unhealthy"

    # Output results
    if json_output:
        result = {
            "status": status,
            "version": __version__,
            "commit": git_sha,
            "checks": checks,
        }
        typer.echo(json_module.dumps(result, indent=2))
    else:
        typer.echo(f"Status: {status}")
        typer.echo(f"Version: {__version__}")
        typer.echo(f"Commit: {git_sha}")

        if verbose:
            typer.echo("\nChecks:")
            for name, info in checks.items():
                check_status = info["status"]
                if check_status == "ok":
                    status_icon = "✓"
                elif check_status in ("warn", "skip"):
                    status_icon = "⚠"
                else:
                    status_icon = "✗"
                typer.echo(f"  {status_icon} {name}: {info['value']}")

    # Exit with appropriate code
    if not overall_healthy:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
