"""CLI helper functions for plugin instantiation and database resolution."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings
    from elspeth.core.landscape.recorder import LandscapeRecorder


def instantiate_plugins_from_config(config: "ElspethSettings") -> dict[str, Any]:
    """Instantiate all plugins from configuration.

    Creates plugin instances BEFORE graph construction,
    enabling schema extraction from instance attributes.

    Args:
        config: Validated ElspethSettings instance

    Returns:
        Dict with keys:
            - source: SourceProtocol instance
            - transforms: list[TransformProtocol] (row_plugins only)
            - sinks: dict[str, SinkProtocol]
            - aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]]

    Raises:
        ValueError: If config references unknown plugins (raised by PluginManager)
    """
    from elspeth.cli import _get_plugin_manager

    manager = _get_plugin_manager()

    # Instantiate source (raises on unknown plugin)
    source_cls = manager.get_source_by_name(config.source.plugin)
    source = source_cls(dict(config.source.options))

    # Instantiate transforms
    transforms = []
    for plugin_config in config.transforms:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        transforms.append(transform_cls(dict(plugin_config.options)))

    # Instantiate aggregations
    # Aggregations REQUIRE batch-aware transforms (is_batch_aware=True).
    # Non-batch-aware transforms process rows individually, ignoring aggregation
    # triggers entirely - a silent misconfiguration that produces wrong results.
    aggregations = {}
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        transform = transform_cls(dict(agg_config.options))

        # Validate batch-aware requirement (fail-fast before graph construction)
        if not getattr(transform, "is_batch_aware", False):
            raise ValueError(
                f"Aggregation '{agg_config.name}' uses transform '{agg_config.plugin}' "
                f"which has is_batch_aware=False. Aggregations require batch-aware "
                f"transforms that can process multiple rows at once. "
                f"Use a batch-aware transform like 'azure_batch_llm', 'batch_stats', "
                f"or 'batch_replicate', or set is_batch_aware=True on your custom transform."
            )

        aggregations[agg_config.name] = (transform, agg_config)

    # Instantiate sinks
    sinks = {}
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        sinks[sink_name] = sink_cls(dict(sink_config.options))

    return {
        "source": source,
        "transforms": transforms,
        "sinks": sinks,
        "aggregations": aggregations,
    }


def resolve_database_url(
    database: str | None,
    settings_path: Path | None,
) -> tuple[str, "ElspethSettings | None"]:
    """Resolve database URL from CLI option or settings file.

    Priority: CLI --database > explicit --settings > settings.yaml landscape.url

    Args:
        database: Explicit database path from CLI (optional)
        settings_path: Path to settings.yaml file (optional)

    Returns:
        Tuple of (database_url, config_or_none)

    Raises:
        ValueError: If database file not found, settings invalid, or neither provided
    """
    from elspeth.core.config import load_settings

    config: ElspethSettings | None = None

    if database:
        db_path = Path(database).expanduser().resolve()
        # Fail fast with clear error if file doesn't exist
        if not db_path.exists():
            raise ValueError(f"Database file not found: {db_path}")
        return f"sqlite:///{db_path}", None

    # Try explicit settings file
    if settings_path is not None:
        if not settings_path.exists():
            raise ValueError(f"Settings file not found: {settings_path}")
        try:
            config = load_settings(settings_path)
            return config.landscape.url, config
        except Exception as e:
            raise ValueError(f"Error loading settings from {settings_path}: {e}") from e

    # Try default settings.yaml - DO NOT silently swallow errors
    default_settings = Path("settings.yaml")
    if default_settings.exists():
        try:
            config = load_settings(default_settings)
            return config.landscape.url, config
        except Exception as e:
            # Don't silently fall through - user should know why settings.yaml failed
            raise ValueError(f"Error loading default settings.yaml: {e}") from e

    raise ValueError("No database specified. Provide --database or ensure settings.yaml exists with landscape.url configured.")


def resolve_latest_run_id(recorder: "LandscapeRecorder") -> str | None:
    """Get the most recently started run ID.

    Args:
        recorder: LandscapeRecorder with database connection

    Returns:
        Run ID of most recent run, or None if no runs exist
    """
    runs = recorder.list_runs()
    if not runs:
        return None
    # list_runs returns ordered by started_at DESC
    return runs[0].run_id


def resolve_run_id(run_id: str, recorder: "LandscapeRecorder") -> str | None:
    """Resolve run_id, handling 'latest' keyword.

    Args:
        run_id: Explicit run ID or 'latest'
        recorder: LandscapeRecorder for looking up latest

    Returns:
        Resolved run ID, or None if 'latest' requested but no runs exist
    """
    if run_id.lower() == "latest":
        return resolve_latest_run_id(recorder)
    return run_id
