# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError

from elspeth import __version__
from elspeth.contracts import ExecutionResult
from elspeth.core.config import ElspethSettings, load_settings, resolve_config
from elspeth.core.dag import ExecutionGraph, GraphValidationError

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import PipelineConfig
    from elspeth.plugins.manager import PluginManager

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


def _execute_pipeline(config: ElspethSettings, verbose: bool = False) -> ExecutionResult:
    """Execute a pipeline from configuration.

    Args:
        config: Validated ElspethSettings instance.
        verbose: Show detailed output.

    Returns:
        ExecutionResult with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseSink, BaseTransform

    # Get plugin manager for dynamic plugin lookup
    manager = _get_plugin_manager()

    # Instantiate source via PluginManager
    source_plugin = config.datasource.plugin
    source_options = dict(config.datasource.options)

    source_cls = manager.get_source_by_name(source_plugin)
    if source_cls is None:
        available = [s.name for s in manager.get_sources()]
        raise ValueError(f"Unknown source plugin: {source_plugin}. Available: {available}")
    source = source_cls(source_options)

    # Instantiate sinks via PluginManager
    sinks: dict[str, BaseSink] = {}
    for sink_name, sink_settings in config.sinks.items():
        sink_plugin = sink_settings.plugin
        sink_options = dict(sink_settings.options)

        sink_cls = manager.get_sink_by_name(sink_plugin)
        if sink_cls is None:
            available = [s.name for s in manager.get_sinks()]
            raise ValueError(f"Unknown sink plugin: {sink_plugin}. Available: {available}")
        sinks[sink_name] = sink_cls(sink_options)  # type: ignore[assignment]

    # Get database URL from settings
    db_url = config.landscape.url
    db = LandscapeDB.from_url(db_url)

    # Instantiate transforms from row_plugins via PluginManager
    transforms: list[BaseTransform] = []
    for plugin_config in config.row_plugins:
        plugin_name = plugin_config.plugin
        plugin_options = dict(plugin_config.options)

        transform_cls = manager.get_transform_by_name(plugin_name)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise typer.BadParameter(f"Unknown transform plugin: {plugin_name}. Available: {available}")
        transforms.append(transform_cls(plugin_options))  # type: ignore[arg-type]

    # Build execution graph from config (needed before PipelineConfig for aggregation node IDs)
    graph = ExecutionGraph.from_config(config)

    # Build aggregation_settings dict (node_id -> AggregationSettings)
    # Also instantiate aggregation transforms and add to transforms list
    from elspeth.core.config import AggregationSettings

    aggregation_settings: dict[str, AggregationSettings] = {}
    agg_id_map = graph.get_aggregation_id_map()
    for agg_config in config.aggregations:
        node_id = agg_id_map[agg_config.name]
        aggregation_settings[node_id] = agg_config

        # Instantiate the aggregation transform plugin via PluginManager
        plugin_name = agg_config.plugin
        transform_cls = manager.get_transform_by_name(plugin_name)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise typer.BadParameter(f"Unknown aggregation plugin: {plugin_name}. Available: {available}")

        # Merge aggregation options with schema from config
        agg_options = dict(agg_config.options)
        transform = transform_cls(agg_options)

        # Set node_id so processor can identify this as an aggregation node
        transform.node_id = node_id

        # Add to transforms list (after row_plugins transforms)
        transforms.append(transform)  # type: ignore[arg-type]

    # Build PipelineConfig with resolved configuration for audit
    # NOTE: Type ignores needed because:
    # - Source plugins implement SourceProtocol structurally but mypy doesn't recognize it
    # - list is invariant so list[BaseTransform] != list[TransformLike]
    # - Sinks implement SinkProtocol structurally but mypy doesn't recognize it
    pipeline_config = PipelineConfig(
        source=source,  # type: ignore[arg-type]
        transforms=transforms,  # type: ignore[arg-type]
        sinks=sinks,  # type: ignore[arg-type]
        config=resolve_config(config),
        gates=list(config.gates),  # Config-driven gates
        aggregation_settings=aggregation_settings,  # Aggregation configurations
    )

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


@dataclass(frozen=True)
class PluginInfo:
    """Metadata for a registered plugin.

    Attributes:
        name: The plugin identifier used in configuration files.
        description: Human-readable description of the plugin's purpose.
    """

    name: str
    description: str


# Registry of built-in plugins (static for Phase 4)
PLUGIN_REGISTRY: dict[str, list[PluginInfo]] = {
    "source": [
        PluginInfo(name="csv", description="Load rows from CSV files"),
        PluginInfo(name="json", description="Load rows from JSON/JSONL files"),
    ],
    "transform": [
        PluginInfo(name="passthrough", description="Pass rows through unchanged"),
        PluginInfo(name="field_mapper", description="Rename, select, and reorganize fields"),
        PluginInfo(name="json_explode", description="Explode array field into multiple rows"),
        PluginInfo(
            name="keyword_filter",
            description="Filter rows containing blocked content patterns",
        ),
        PluginInfo(
            name="azure_content_safety",
            description="Azure Content Safety API for hate, violence, sexual, self-harm detection",
        ),
        PluginInfo(
            name="azure_prompt_shield",
            description="Azure Prompt Shield API for jailbreak and prompt injection detection",
        ),
    ],
    "sink": [
        PluginInfo(name="csv", description="Write rows to CSV files"),
        PluginInfo(name="json", description="Write rows to JSON/JSONL files"),
        PluginInfo(name="database", description="Write rows to database tables"),
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
    valid_types = set(PLUGIN_REGISTRY.keys())

    if plugin_type and plugin_type not in valid_types:
        typer.echo(f"Error: Invalid type '{plugin_type}'.", err=True)
        typer.echo(f"Valid types: {', '.join(sorted(valid_types))}", err=True)
        raise typer.Exit(1)

    types_to_show = [plugin_type] if plugin_type else list(PLUGIN_REGISTRY.keys())

    for ptype in types_to_show:
        # types_to_show only contains keys from PLUGIN_REGISTRY (either filtered by validated plugin_type
        # or directly from PLUGIN_REGISTRY.keys()), so direct access is safe
        plugins = PLUGIN_REGISTRY[ptype]
        if plugins:
            typer.echo(f"\n{ptype.upper()}S:")
            for plugin in plugins:
                typer.echo(f"  {plugin.name:12} - {plugin.description}")
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
    retention_days: int = typer.Option(
        90,
        "--retention-days",
        "-r",
        help="Delete payloads older than this many days.",
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

    try:
        payload_store = FilesystemPayloadStore(payload_path)
        purge_manager = PurgeManager(db, payload_store)

        # Find expired payloads
        expired_refs = purge_manager.find_expired_row_payloads(retention_days)

        if not expired_refs:
            typer.echo(f"No payloads older than {retention_days} days found.")
            return

        if dry_run:
            typer.echo(f"Would delete {len(expired_refs)} payload(s) older than {retention_days} days:")
            for ref in expired_refs[:10]:  # Show first 10
                exists = payload_store.exists(ref)
                status = "exists" if exists else "already deleted"
                typer.echo(f"  {ref[:16]}... ({status})")
            if len(expired_refs) > 10:
                typer.echo(f"  ... and {len(expired_refs) - 10} more")
            return

        # Confirm unless --yes
        if not yes:
            confirm = typer.confirm(f"Delete {len(expired_refs)} payload(s) older than {retention_days} days?")
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


def _build_resume_pipeline_config(
    settings: ElspethSettings,
) -> PipelineConfig:
    """Build PipelineConfig for resume from settings.

    For resume, source is NullSource (data comes from payloads).
    Transforms and sinks are rebuilt from settings.

    Args:
        settings: Full ElspethSettings configuration.

    Returns:
        PipelineConfig ready for resume.
    """
    from elspeth.engine import PipelineConfig
    from elspeth.plugins.base import BaseSink, BaseTransform
    from elspeth.plugins.llm.azure import AzureLLMTransform
    from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
    from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.sinks.database_sink import DatabaseSink
    from elspeth.plugins.sinks.json_sink import JSONSink
    from elspeth.plugins.sources.null_source import NullSource
    from elspeth.plugins.transforms import FieldMapper, PassThrough
    from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
    from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
    from elspeth.plugins.transforms.batch_replicate import BatchReplicate
    from elspeth.plugins.transforms.batch_stats import BatchStats
    from elspeth.plugins.transforms.json_explode import JSONExplode
    from elspeth.plugins.transforms.keyword_filter import KeywordFilter

    # Plugin registries (same as _execute_pipeline)
    TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
        "passthrough": PassThrough,
        "field_mapper": FieldMapper,
        "batch_stats": BatchStats,
        "json_explode": JSONExplode,
        "keyword_filter": KeywordFilter,
        "azure_content_safety": AzureContentSafety,
        "azure_prompt_shield": AzurePromptShield,
        "batch_replicate": BatchReplicate,
        "openrouter_llm": OpenRouterLLMTransform,
        "azure_llm": AzureLLMTransform,
        "azure_batch_llm": AzureBatchLLMTransform,
    }

    # Source is NullSource for resume - data comes from payloads
    source = NullSource({})

    # Build transforms from settings (same logic as _execute_pipeline)
    transforms: list[BaseTransform] = []
    for plugin_config in settings.row_plugins:
        plugin_name = plugin_config.plugin
        plugin_options = dict(plugin_config.options)

        if plugin_name not in TRANSFORM_PLUGINS:
            raise ValueError(f"Unknown transform plugin: {plugin_name}")
        transform_class = TRANSFORM_PLUGINS[plugin_name]
        transforms.append(transform_class(plugin_options))

    # Build aggregation transforms (same logic as _execute_pipeline)
    # Need the graph to get aggregation node IDs
    graph = ExecutionGraph.from_config(settings)
    agg_id_map = graph.get_aggregation_id_map()

    from elspeth.core.config import AggregationSettings

    aggregation_settings: dict[str, AggregationSettings] = {}
    for agg_config in settings.aggregations:
        node_id = agg_id_map[agg_config.name]
        aggregation_settings[node_id] = agg_config

        plugin_name = agg_config.plugin
        if plugin_name not in TRANSFORM_PLUGINS:
            raise ValueError(f"Unknown aggregation plugin: {plugin_name}")
        transform_class = TRANSFORM_PLUGINS[plugin_name]

        agg_options = dict(agg_config.options)
        transform = transform_class(agg_options)
        transform.node_id = node_id
        transforms.append(transform)

    # Build sinks from settings
    # CRITICAL: Resume must append to existing output, not overwrite
    sinks: dict[str, BaseSink] = {}
    for sink_name, sink_settings in settings.sinks.items():
        sink_plugin = sink_settings.plugin
        sink_options = dict(sink_settings.options)
        sink_options["mode"] = "append"  # Resume appends to existing files

        sink: BaseSink
        if sink_plugin == "csv":
            sink = CSVSink(sink_options)
        elif sink_plugin == "json":
            sink = JSONSink(sink_options)
        elif sink_plugin == "database":
            sink = DatabaseSink(sink_options)
        else:
            raise ValueError(f"Unknown sink plugin: {sink_plugin}")

        sinks[sink_name] = sink

    return PipelineConfig(
        source=source,  # type: ignore[arg-type]
        transforms=transforms,  # type: ignore[arg-type]
        sinks=sinks,  # type: ignore[arg-type]
        config=resolve_config(settings),
        gates=list(settings.gates),
        aggregation_settings=aggregation_settings,
    )


def _build_resume_graph_from_db(
    db: LandscapeDB,
    run_id: str,
) -> ExecutionGraph:
    """Reconstruct ExecutionGraph from nodes/edges registered in database.

    Args:
        db: LandscapeDB connection.
        run_id: Run ID to reconstruct graph for.

    Returns:
        ExecutionGraph reconstructed from database.
    """
    import json

    from sqlalchemy import select

    from elspeth.core.landscape import edges_table, nodes_table

    graph = ExecutionGraph()

    with db.engine.connect() as conn:
        nodes = conn.execute(select(nodes_table).where(nodes_table.c.run_id == run_id)).fetchall()

        edges = conn.execute(select(edges_table).where(edges_table.c.run_id == run_id)).fetchall()

    for node in nodes:
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            plugin_name=node.plugin_name,
            config=json.loads(node.config_json) if node.config_json else {},
        )

    for edge in edges:
        graph.add_edge(edge.from_node_id, edge.to_node_id, label=edge.label)

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

    # Try to load settings - needed for execute mode and optional for dry-run
    settings_config: ElspethSettings | None = None
    settings_path = Path(settings_file) if settings_file else Path("settings.yaml")
    if settings_path.exists():
        try:
            settings_config = load_settings(settings_path)
        except Exception as e:
            if execute:
                typer.echo(f"Error loading {settings_path}: {e}", err=True)
                typer.echo(
                    "Settings are required for --execute mode to rebuild pipeline.",
                    err=True,
                )
                raise typer.Exit(1) from None
            # For dry-run, settings are optional - continue without

    # Resolve database URL
    db_url: str | None = None

    if database:
        db_path = Path(database)
        db_url = f"sqlite:///{db_path.resolve()}"
    elif settings_config is not None:
        db_url = settings_config.landscape.url
        typer.echo(f"Using database from settings.yaml: {db_url}")
    else:
        typer.echo("Error: No settings.yaml found and --database not provided.", err=True)
        typer.echo("Specify --database to provide path to Landscape database.", err=True)
        raise typer.Exit(1)

    # Initialize database and recovery manager
    try:
        db = LandscapeDB.from_url(db_url)
    except Exception as e:
        typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        checkpoint_manager = CheckpointManager(db)
        recovery_manager = RecoveryManager(db, checkpoint_manager)

        # Check if run can be resumed
        check = recovery_manager.can_resume(run_id)

        if not check.can_resume:
            typer.echo(f"Cannot resume run {run_id}: {check.reason}", err=True)
            raise typer.Exit(1)

        # Get resume point information
        resume_point = recovery_manager.get_resume_point(run_id)
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
            return

        # Execute resume
        if settings_config is None:
            typer.echo("Error: settings.yaml required for --execute mode.", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nResuming run {run_id}...")

        # Get payload store from settings
        from elspeth.core.payload_store import FilesystemPayloadStore

        payload_path = settings_config.payload_store.base_path
        if not payload_path.exists():
            typer.echo(f"Error: Payload directory not found: {payload_path}", err=True)
            raise typer.Exit(1)

        payload_store = FilesystemPayloadStore(payload_path)

        # Build pipeline config and graph for resume
        pipeline_config = _build_resume_pipeline_config(settings_config)
        graph = _build_resume_graph_from_db(db, run_id)

        # Create orchestrator with checkpoint manager and resume
        from elspeth.engine import Orchestrator

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_manager)

        try:
            result = orchestrator.resume(
                resume_point=resume_point,
                config=pipeline_config,
                graph=graph,
                payload_store=payload_store,
                settings=settings_config,
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


if __name__ == "__main__":
    app()
