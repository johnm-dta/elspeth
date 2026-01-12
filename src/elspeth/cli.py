# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

import typer

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
    typer.echo(f"Validate command not yet implemented. Settings: {settings}")
    raise typer.Exit(1)


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
