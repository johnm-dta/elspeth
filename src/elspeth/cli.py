# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from elspeth import __version__
from elspeth.core.config import ElspethSettings, load_settings
from elspeth.core.dag import ExecutionGraph, GraphValidationError

app = typer.Typer(
    name="elspeth",
    help="ELSPETH: Auditable Sense/Decide/Act pipelines.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"elspeth version {__version__}")
        raise typer.Exit()


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
) -> None:
    """ELSPETH: Auditable Sense/Decide/Act pipelines."""
    pass


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
) -> None:
    """Execute a pipeline run.

    Requires --execute flag to actually run (safety feature).
    Use --dry-run to validate configuration without executing.
    """
    settings_path = Path(settings)

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1) from None

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    if verbose:
        typer.echo(f"Graph validated: {graph.node_count} nodes, {graph.edge_count} edges")

    if dry_run:
        typer.echo("Dry run mode - would execute:")
        typer.echo(f"  Source: {config.datasource.plugin}")
        typer.echo(f"  Transforms: {len(config.row_plugins)}")
        typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
        typer.echo(f"  Output sink: {config.output_sink}")
        if verbose:
            typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
            typer.echo(f"  Execution order: {len(graph.topological_order())} steps")
            typer.echo(f"  Concurrency: {config.concurrency.max_workers} workers")
            typer.echo(f"  Landscape: {config.landscape.url}")
        return

    # Safety check: require explicit --execute flag
    if not execute:
        typer.echo("Pipeline configuration valid.")
        typer.echo(f"  Source: {config.datasource.plugin}")
        typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
        typer.echo("")
        typer.echo("To execute, add --execute (or -x) flag:", err=True)
        typer.echo(f"  elspeth run -s {settings} --execute", err=True)
        raise typer.Exit(1)

    # Execute pipeline with validated config
    try:
        result = _execute_pipeline(config, verbose=verbose)
        typer.echo(f"\nRun completed: {result['status']}")
        typer.echo(f"  Rows processed: {result['rows_processed']}")
        typer.echo(f"  Run ID: {result['run_id']}")
    except Exception as e:
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
        help="Row ID or index to explain.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
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
    """
    import json as json_module

    from elspeth.tui.explain_app import ExplainApp

    # For now, we need a database connection to query
    # This will be integrated with actual runs once Phase 3 is complete

    if json_output:
        # JSON output mode
        result = {
            "run_id": run_id,
            "status": "error",
            "message": "No runs found. Execute 'elspeth run --execute' first.",
        }
        typer.echo(json_module.dumps(result, indent=2))
        raise typer.Exit(1)

    if no_tui:
        # Text output mode
        typer.echo(f"Error: Run '{run_id}' not found.")
        typer.echo("Execute 'elspeth run --execute' to create a run first.")
        raise typer.Exit(1)

    # TUI mode
    tui_app = ExplainApp(
        run_id=run_id if run_id != "latest" else None,
        token_id=token,
        row_id=row,
    )
    tui_app.run()


def _execute_pipeline(config: ElspethSettings, verbose: bool = False) -> dict[str, Any]:
    """Execute a pipeline from configuration.

    Args:
        config: Validated ElspethSettings instance.
        verbose: Show detailed output.

    Returns:
        Dict with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.engine.adapters import SinkAdapter
    from elspeth.plugins.base import BaseGate, BaseSink, BaseSource, BaseTransform
    from elspeth.plugins.gates import FieldMatchGate, FilterGate, ThresholdGate
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.sinks.database_sink import DatabaseSink
    from elspeth.plugins.sinks.json_sink import JSONSink
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.sources.json_source import JSONSource
    from elspeth.plugins.transforms import FieldMapper, PassThrough

    # Plugin registries
    TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
        "passthrough": PassThrough,
        "field_mapper": FieldMapper,
    }
    GATE_PLUGINS: dict[str, type[BaseGate]] = {
        "threshold_gate": ThresholdGate,
        "field_match_gate": FieldMatchGate,
        "filter_gate": FilterGate,
    }

    # Instantiate source from new schema
    source_plugin = config.datasource.plugin
    source_options = dict(config.datasource.options)

    source: BaseSource
    if source_plugin == "csv":
        source = CSVSource(source_options)
    elif source_plugin == "json":
        source = JSONSource(source_options)
    else:
        raise ValueError(f"Unknown source plugin: {source_plugin}")

    # Instantiate sinks and wrap in SinkAdapter for Phase 3B compatibility
    sinks: dict[str, SinkAdapter] = {}
    for sink_name, sink_settings in config.sinks.items():
        sink_plugin = sink_settings.plugin
        sink_options = dict(sink_settings.options)

        raw_sink: BaseSink
        artifact_descriptor: dict[str, Any]
        if sink_plugin == "csv":
            raw_sink = CSVSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "json":
            raw_sink = JSONSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "database":
            raw_sink = DatabaseSink(sink_options)
            artifact_descriptor = {
                "kind": "database",
                "url": sink_options.get("url", ""),
                "table": sink_options.get("table", ""),
            }
        else:
            raise ValueError(f"Unknown sink plugin: {sink_plugin}")

        # Wrap Phase 2 sink in adapter for Phase 3B SinkLike interface
        sinks[sink_name] = SinkAdapter(
            raw_sink,
            plugin_name=sink_plugin,
            sink_name=sink_name,
            artifact_descriptor=artifact_descriptor,
        )

    # Get database URL from settings
    db_url = config.landscape.url
    db = LandscapeDB.from_url(db_url)

    # Instantiate transforms/gates from row_plugins
    transforms: list[BaseTransform | BaseGate] = []
    for plugin_config in config.row_plugins:
        plugin_name = plugin_config.plugin
        plugin_options = dict(plugin_config.options)

        if plugin_config.type == "gate":
            if plugin_name not in GATE_PLUGINS:
                raise typer.BadParameter(f"Unknown gate plugin: {plugin_name}")
            gate_class = GATE_PLUGINS[plugin_name]
            transforms.append(gate_class(plugin_options))
        else:
            if plugin_name not in TRANSFORM_PLUGINS:
                raise typer.BadParameter(f"Unknown transform plugin: {plugin_name}")
            transform_class = TRANSFORM_PLUGINS[plugin_name]
            transforms.append(transform_class(plugin_options))

    # Build PipelineConfig
    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,
        sinks=sinks,
    )

    # Build execution graph from config
    graph = ExecutionGraph.from_config(config)

    if verbose:
        typer.echo("Starting pipeline execution...")

    # Execute via Orchestrator (creates full audit trail)
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config, graph=graph, settings=config)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }


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
    settings_path = Path(settings)

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1) from None

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Configuration valid: {settings_path.name}")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Output: {config.output_sink}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")


# Plugins subcommand group
plugins_app = typer.Typer(help="Plugin management commands.")
app.add_typer(plugins_app, name="plugins")


# Registry of built-in plugins (static for Phase 4)
PLUGIN_REGISTRY = {
    "source": [
        ("csv", "Load rows from CSV files"),
        ("json", "Load rows from JSON/JSONL files"),
    ],
    "transform": [
        ("passthrough", "Pass rows through unchanged"),
        ("field_mapper", "Rename, select, and reorganize fields"),
    ],
    "gate": [
        ("threshold_gate", "Route rows based on numeric threshold"),
        ("field_match_gate", "Route rows based on field value matching"),
        ("filter_gate", "Filter rows based on field conditions"),
    ],
    "sink": [
        ("csv", "Write rows to CSV files"),
        ("json", "Write rows to JSON/JSONL files"),
        ("database", "Write rows to database tables"),
    ],
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
    valid_types = {"source", "transform", "sink"}

    if plugin_type and plugin_type not in valid_types:
        typer.echo(f"Error: Invalid type '{plugin_type}'.", err=True)
        typer.echo(f"Valid types: {', '.join(sorted(valid_types))}", err=True)
        raise typer.Exit(1)

    types_to_show = [plugin_type] if plugin_type else list(PLUGIN_REGISTRY.keys())

    for ptype in types_to_show:
        plugins = PLUGIN_REGISTRY.get(ptype, [])
        if plugins:
            typer.echo(f"\n{ptype.upper()}S:")
            for name, description in plugins:
                typer.echo(f"  {name:12} - {description}")
        elif ptype in valid_types:
            typer.echo(f"\n{ptype.upper()}S:")
            typer.echo("  (none available)")

    typer.echo()  # Final newline


@app.command()
def purge(
    database: str | None = typer.Option(
        None,
        "--database", "-d",
        help="Path to Landscape database file (SQLite).",
    ),
    payload_dir: str | None = typer.Option(
        None,
        "--payload-dir", "-p",
        help="Path to payload storage directory.",
    ),
    retention_days: int = typer.Option(
        90,
        "--retention-days", "-r",
        help="Delete payloads older than this many days.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without deleting.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
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

    if database:
        db_path = Path(database)
        db_url = f"sqlite:///{db_path.resolve()}"
    else:
        # Try loading from settings.yaml
        settings_path = Path("settings.yaml")
        if settings_path.exists():
            try:
                config = load_settings(settings_path)
                db_url = config.landscape.url
                typer.echo(f"Using database from settings.yaml: {db_url}")
            except Exception as e:
                typer.echo(f"Error loading settings.yaml: {e}", err=True)
                typer.echo("Specify --database to provide path directly.", err=True)
                raise typer.Exit(1) from None
        else:
            typer.echo("Error: No settings.yaml found and --database not provided.", err=True)
            typer.echo("Specify --database to provide path to Landscape database.", err=True)
            raise typer.Exit(1) from None

    if payload_dir:
        payload_path = Path(payload_dir)
    else:
        # Default to ./payloads relative to database location
        if database:
            payload_path = Path(database).parent / "payloads"
        else:
            payload_path = Path("payloads")

    # Initialize database and payload store
    try:
        db = LandscapeDB.from_url(db_url)
    except Exception as e:
        typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    payload_store = FilesystemPayloadStore(payload_path)
    purge_manager = PurgeManager(db, payload_store)

    # Find expired payloads
    expired_refs = purge_manager.find_expired_row_payloads(retention_days)

    if not expired_refs:
        typer.echo(f"No payloads older than {retention_days} days found.")
        db.close()
        return

    if dry_run:
        typer.echo(f"Would delete {len(expired_refs)} payload(s) older than {retention_days} days:")
        for ref in expired_refs[:10]:  # Show first 10
            exists = payload_store.exists(ref)
            status = "exists" if exists else "already deleted"
            typer.echo(f"  {ref[:16]}... ({status})")
        if len(expired_refs) > 10:
            typer.echo(f"  ... and {len(expired_refs) - 10} more")
        db.close()
        return

    # Confirm unless --yes
    if not yes:
        confirm = typer.confirm(
            f"Delete {len(expired_refs)} payload(s) older than {retention_days} days?"
        )
        if not confirm:
            typer.echo("Aborted.")
            db.close()
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

    db.close()


if __name__ == "__main__":
    app()
