# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

from pathlib import Path

import typer
import yaml

from elspeth import __version__

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
) -> None:
    """Execute a pipeline run."""
    typer.echo(f"Run command not yet implemented. Settings: {settings}")
    raise typer.Exit(1)


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
) -> None:
    """Explain lineage for a row or token."""
    typer.echo(f"Explain command not yet implemented. Run: {run_id}")
    raise typer.Exit(1)


# Known plugins for validation
KNOWN_SOURCES = {"csv", "json"}
KNOWN_SINKS = {"csv", "json", "database"}


def _validate_config(config: dict) -> list[str]:
    """Validate pipeline configuration structure.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check source
    if "source" not in config:
        errors.append("Missing required 'source' section")
    else:
        source = config["source"]
        if "plugin" not in source:
            errors.append("Source missing 'plugin' field")
        elif source["plugin"] not in KNOWN_SOURCES:
            errors.append(f"Unknown source plugin: {source['plugin']}")

    # Check sinks
    if "sinks" not in config:
        errors.append("Missing required 'sinks' section")
    elif not config["sinks"]:
        errors.append("At least one sink is required")
    else:
        for sink_name, sink_config in config["sinks"].items():
            if "plugin" not in sink_config:
                errors.append(f"Sink '{sink_name}' missing 'plugin' field")
            elif sink_config["plugin"] not in KNOWN_SINKS:
                errors.append(f"Unknown sink plugin: {sink_config['plugin']}")

    return errors


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

    # Check file exists
    if not settings_path.exists():
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)

    # Parse YAML
    try:
        with open(settings_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        typer.echo(f"Error: Invalid YAML: {e}", err=True)
        raise typer.Exit(1)

    # Validate structure
    errors = _validate_config(config)
    if errors:
        typer.echo("Configuration errors:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration valid: {settings_path.name}")
    typer.echo(f"  Source: {config['source']['plugin']}")
    typer.echo(f"  Sinks: {', '.join(config['sinks'].keys())}")


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
        # No built-in transforms in Phase 4
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


if __name__ == "__main__":
    app()
