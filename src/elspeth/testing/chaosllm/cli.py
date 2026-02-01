# src/elspeth/testing/chaosllm/cli.py
"""CLI for ChaosLLM fake LLM server.

Provides command-line interface for starting and managing the ChaosLLM
server for load testing and fault injection scenarios.

Usage:
    # Start server with defaults
    chaosllm serve

    # Start with a preset
    chaosllm serve --preset=stress_aimd

    # Start with custom configuration
    chaosllm serve --config=my_chaos.yaml --port=9000

    # Override specific error rates
    chaosllm serve --rate-limit-pct=10 --capacity-529-pct=5

    # Start MCP server for metrics analysis
    chaosllm-mcp --database=./chaosllm-metrics.db
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from elspeth.testing.chaosllm.config import (
    DEFAULT_MEMORY_DB,
    list_presets,
    load_config,
)

# Main CLI app
app = typer.Typer(
    name="chaosllm",
    help="ChaosLLM: Fake LLM server for load testing and fault injection.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        # Use elspeth version
        try:
            from elspeth import __version__

            typer.echo(f"chaosllm (elspeth {__version__})")
        except ImportError:
            typer.echo("chaosllm (version unknown)")
        raise typer.Exit()


@app.command()
def serve(
    # Configuration sources
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            "-p",
            help="Preset configuration to use. Use 'chaosllm presets' to list available presets.",
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    # Server binding
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host address to bind to.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-P",
            help="Port to listen on.",
            min=1,
            max=65535,
        ),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of uvicorn workers.",
            min=1,
        ),
    ] = 1,
    # Metrics database
    database: Annotated[
        str,
        typer.Option(
            "--database",
            "-d",
            help="SQLite database path for metrics storage.",
        ),
    ] = DEFAULT_MEMORY_DB,
    # Error injection overrides
    rate_limit_pct: Annotated[
        float | None,
        typer.Option(
            "--rate-limit-pct",
            help="429 Rate Limit error percentage (0-100).",
            min=0.0,
            max=100.0,
        ),
    ] = None,
    capacity_529_pct: Annotated[
        float | None,
        typer.Option(
            "--capacity-529-pct",
            help="529 Capacity error percentage (0-100).",
            min=0.0,
            max=100.0,
        ),
    ] = None,
    service_unavailable_pct: Annotated[
        float | None,
        typer.Option(
            "--service-unavailable-pct",
            help="503 Service Unavailable error percentage (0-100).",
            min=0.0,
            max=100.0,
        ),
    ] = None,
    internal_error_pct: Annotated[
        float | None,
        typer.Option(
            "--internal-error-pct",
            help="500 Internal Error percentage (0-100).",
            min=0.0,
            max=100.0,
        ),
    ] = None,
    timeout_pct: Annotated[
        float | None,
        typer.Option(
            "--timeout-pct",
            help="Connection timeout percentage (0-100).",
            min=0.0,
            max=100.0,
        ),
    ] = None,
    selection_mode: Annotated[
        str | None,
        typer.Option(
            "--selection-mode",
            help="Error selection strategy: priority or weighted.",
        ),
    ] = None,
    # Latency overrides
    base_ms: Annotated[
        int | None,
        typer.Option(
            "--base-ms",
            help="Base latency in milliseconds.",
            min=0,
        ),
    ] = None,
    jitter_ms: Annotated[
        int | None,
        typer.Option(
            "--jitter-ms",
            help="Latency jitter in milliseconds.",
            min=0,
        ),
    ] = None,
    # Response mode
    response_mode: Annotated[
        str | None,
        typer.Option(
            "--response-mode",
            help="Response generation mode: random, template, echo, preset.",
        ),
    ] = None,
    # Burst settings
    burst_enabled: Annotated[
        bool | None,
        typer.Option(
            "--burst-enabled/--no-burst",
            help="Enable burst pattern injection.",
        ),
    ] = None,
    burst_interval_sec: Annotated[
        int | None,
        typer.Option(
            "--burst-interval-sec",
            help="Time between burst starts in seconds.",
            min=1,
        ),
    ] = None,
    burst_duration_sec: Annotated[
        int | None,
        typer.Option(
            "--burst-duration-sec",
            help="How long each burst lasts in seconds.",
            min=1,
        ),
    ] = None,
    # Misc options
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Start the ChaosLLM fake LLM server.

    The server provides OpenAI and Azure OpenAI compatible endpoints with
    configurable error injection, latency simulation, and response generation.

    Configuration precedence (highest to lowest):
    1. Command-line flags
    2. Config file (--config)
    3. Preset (--preset)
    4. Built-in defaults

    Examples:

        # Start with defaults
        chaosllm serve

        # Use a preset
        chaosllm serve --preset=stress_aimd

        # Custom error rates
        chaosllm serve --rate-limit-pct=10 --capacity-529-pct=5

        # Custom port and database
        chaosllm serve --port=9000 --database=./my-metrics.db
    """
    # Build CLI overrides dict
    cli_overrides: dict[str, Any] = {
        "server": {
            "host": host,
            "port": port,
            "workers": workers,
        },
        "metrics": {
            "database": database,
        },
    }

    # Collect error injection overrides
    error_overrides: dict[str, Any] = {}
    if rate_limit_pct is not None:
        error_overrides["rate_limit_pct"] = rate_limit_pct
    if capacity_529_pct is not None:
        error_overrides["capacity_529_pct"] = capacity_529_pct
    if service_unavailable_pct is not None:
        error_overrides["service_unavailable_pct"] = service_unavailable_pct
    if internal_error_pct is not None:
        error_overrides["internal_error_pct"] = internal_error_pct
    if timeout_pct is not None:
        error_overrides["timeout_pct"] = timeout_pct
    if selection_mode is not None:
        error_overrides["selection_mode"] = selection_mode

    # Burst overrides
    burst_overrides: dict[str, Any] = {}
    if burst_enabled is not None:
        burst_overrides["enabled"] = burst_enabled
    if burst_interval_sec is not None:
        burst_overrides["interval_sec"] = burst_interval_sec
    if burst_duration_sec is not None:
        burst_overrides["duration_sec"] = burst_duration_sec

    if burst_overrides:
        error_overrides["burst"] = burst_overrides

    if error_overrides:
        cli_overrides["error_injection"] = error_overrides

    # Latency overrides
    latency_overrides: dict[str, int] = {}
    if base_ms is not None:
        latency_overrides["base_ms"] = base_ms
    if jitter_ms is not None:
        latency_overrides["jitter_ms"] = jitter_ms

    if latency_overrides:
        cli_overrides["latency"] = latency_overrides

    # Response overrides
    if response_mode is not None:
        cli_overrides["response"] = {"mode": response_mode}

    # Load configuration
    try:
        config = load_config(
            preset=preset,
            config_file=config_file,
            cli_overrides=cli_overrides,
        )
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    # Print startup info
    typer.secho(
        f"Starting ChaosLLM server on {config.server.host}:{config.server.port}",
        fg=typer.colors.GREEN,
    )
    if preset:
        typer.echo(f"  Preset: {preset}")
    if config_file:
        typer.echo(f"  Config: {config_file}")
    typer.echo(f"  Metrics DB: {config.metrics.database}")
    typer.echo(f"  Workers: {config.server.workers}")

    # Show error injection summary
    error_cfg = config.error_injection
    active_errors = []
    if error_cfg.rate_limit_pct > 0:
        active_errors.append(f"429:{error_cfg.rate_limit_pct:.1f}%")
    if error_cfg.capacity_529_pct > 0:
        active_errors.append(f"529:{error_cfg.capacity_529_pct:.1f}%")
    if error_cfg.service_unavailable_pct > 0:
        active_errors.append(f"503:{error_cfg.service_unavailable_pct:.1f}%")
    if error_cfg.internal_error_pct > 0:
        active_errors.append(f"500:{error_cfg.internal_error_pct:.1f}%")

    if active_errors:
        typer.echo(f"  Error injection: {', '.join(active_errors)}")
    else:
        typer.echo("  Error injection: disabled")

    if error_cfg.burst.enabled:
        typer.echo(f"  Burst mode: every {error_cfg.burst.interval_sec}s for {error_cfg.burst.duration_sec}s")

    typer.echo()

    # Start the server with uvicorn
    try:
        import uvicorn
    except ImportError as e:
        typer.secho(
            "Error: uvicorn is not installed. Install with: uv pip install uvicorn",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from e

    from elspeth.testing.chaosllm.server import create_app

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level="info",
    )


@app.command()
def presets() -> None:
    """List available preset configurations.

    Presets provide pre-configured error injection and response patterns
    for common testing scenarios.
    """
    available = list_presets()

    if not available:
        typer.echo("No presets found.")
        return

    typer.secho("Available presets:", fg=typer.colors.GREEN)
    for name in sorted(available):
        typer.echo(f"  - {name}")

    typer.echo()
    typer.echo("Use with: chaosllm serve --preset=<name>")


@app.command()
def show_config(
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            "-p",
            help="Preset to show configuration for.",
        ),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Config file to show.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json or yaml.",
        ),
    ] = "yaml",
) -> None:
    """Show the effective configuration.

    Displays the merged configuration from preset and/or config file.
    """
    try:
        config = load_config(
            preset=preset,
            config_file=config_file,
        )
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.secho(f"Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    config_dict = config.model_dump()

    if output_format == "json":
        typer.echo(json.dumps(config_dict, indent=2))
    else:
        # YAML output
        try:
            import yaml

            typer.echo(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        except ImportError:
            # Fall back to JSON if yaml not available
            typer.echo(json.dumps(config_dict, indent=2))


# MCP server CLI - separate entry point
mcp_app = typer.Typer(
    name="chaosllm-mcp",
    help="ChaosLLM MCP server for metrics analysis.",
    no_args_is_help=False,
)


@mcp_app.callback(invoke_without_command=True)
def mcp_main(
    database: Annotated[
        str | None,
        typer.Option(
            "--database",
            "-d",
            help="SQLite database path. If not specified, searches for chaosllm-metrics.db.",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Start the ChaosLLM MCP server for metrics analysis.

    The MCP server provides Claude-optimized tools for analyzing ChaosLLM
    metrics and investigating error patterns.

    If no database is specified, searches for chaosllm-metrics.db in the
    current directory and subdirectories.
    """
    # Find database
    if database is None:
        # Search for default database
        candidates = [
            Path("./chaosllm-metrics.db"),
            Path("./runs/chaosllm-metrics.db"),
        ]

        for candidate in candidates:
            if candidate.exists():
                database = str(candidate)
                break

        if database is None:
            typer.secho(
                "Error: No database found. Specify with --database or create one by running 'chaosllm serve' first.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

    db_path = Path(database)
    if not db_path.exists():
        typer.secho(
            f"Error: Database not found: {database}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    typer.secho(f"Starting ChaosLLM MCP server with database: {database}", fg=typer.colors.GREEN)

    # Import and start the MCP server
    # The chaosllm_mcp module is implemented in a separate task
    try:
        # Type ignore because module may not exist yet during development
        import elspeth.testing.chaosllm_mcp.server as mcp_server  # type: ignore[import-not-found]

        mcp_server.serve(database)  # type: ignore[attr-defined]
    except ImportError as e:
        typer.secho(
            f"Error: MCP server not available. The chaosllm_mcp module may not be installed yet.\n{e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from e


def main() -> None:
    """Entry point for chaosllm CLI."""
    app()


def mcp_main_entry() -> None:
    """Entry point for chaosllm-mcp CLI."""
    mcp_app()


if __name__ == "__main__":
    main()
