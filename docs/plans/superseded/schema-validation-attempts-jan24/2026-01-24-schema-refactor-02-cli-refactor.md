# Schema Validation Refactor - CLI Refactor Tasks (5-7)

> **Previous:** `01-foundation.md` | **Next:** `03-testing.md`

This file contains CLI command refactoring to use the new graph construction flow.

## Spec Corrections (Post-Implementation)

**NOTE:** The following corrections were made during implementation after verifying actual codebase APIs:

### Task 7: Orchestrator API Correction

**Original spec** described a simplified Orchestrator API that does not exist:
```python
# SPEC (fictional API - does not exist)
orchestrator = Orchestrator(
    pipeline_config=pipeline_config,
    landscape_db=db,
    event_bus=event_bus,
    graph=graph,
    resume_run_id=run_id,
)
result = orchestrator.resume(run_id)
```

**Actual implementation** uses the correct Orchestrator API from `src/elspeth/engine/orchestrator.py`:
```python
# ACTUAL (correct API)
checkpoint_manager = CheckpointManager(db)
orchestrator = Orchestrator(db, event_bus=event_bus, checkpoint_manager=checkpoint_manager)
result = orchestrator.resume(
    resume_point=resume_point,
    config=pipeline_config,
    graph=graph,
    payload_store=payload_store,
    settings=config,
)
```

**Key differences:**
1. `Orchestrator.__init__` signature: `(db, *, event_bus=None, checkpoint_manager=None, ...)` not `(pipeline_config, landscape_db, event_bus, graph, resume_run_id)`
2. `orchestrator.resume()` signature: `(resume_point, config, graph, *, payload_store=None, settings=None)` not `(run_id)`
3. EventBus and CheckpointManager are **not mutually exclusive** - both can be passed to constructor

**EventBus Addition (Post-Review):**
Architecture review identified missing EventBus in resume command as High severity UX bug. Added in commit bea9bba:
- Resume now creates EventBus with console formatters (matching run command)
- Users get real-time progress feedback during resume operations
- EventBus coexists with CheckpointManager (both passed to Orchestrator)

---

## Task 5: Refactor CLI `run()` Command + Add `_execute_pipeline_with_instances()`

**Files:**
- Modify: `src/elspeth/cli.py` (run command + new execution helper)
- Test: `tests/integration/test_cli_schema_validation.py`

**Purpose:** Use new graph construction in run command. **Complete implementation** of execution helper (no truncation).

### Step 1: Write failing integration test

**File:** `tests/integration/test_cli_schema_validation.py`

```python
"""Integration tests for CLI schema validation."""

import tempfile
from pathlib import Path
from typer.testing import CliRunner
from elspeth.cli import app


def test_cli_run_detects_schema_incompatibility():
    """Verify CLI run command detects schema incompatibility."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        field_a: {type: str}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          field_b: {type: int}  # INCOMPATIBLE

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["run", "--settings", str(config_file)])

        # Should fail with schema error
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_run_detects_schema_incompatibility -v`

Expected: FAIL (CLI still uses old construction)

### Step 3: Refactor run() command

**File:** `src/elspeth/cli.py` (modify run command, ~lines 156-223)

```python
@app.command()
def run(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Validate without executing."),
    execute: bool = typer.Option(False, "--execute", "-x", help="Actually execute (required)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output."),
    output_format: Literal["console", "json"] = typer.Option("console", "--format", "-f"),
) -> None:
    """Execute a pipeline run.

    Requires --execute flag to actually run (safety feature).
    Use --dry-run to validate configuration without executing.
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    settings_path = Path(settings).expanduser()

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
            output_sink=config.output_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    # Console messages (skip in JSON mode)
    if output_format == "console":
        if verbose:
            typer.echo(f"Graph validated: {graph.node_count} nodes, {graph.edge_count} edges")

        if dry_run:
            typer.echo("Dry run mode - would execute:")
            typer.echo(f"  Source: {config.datasource.plugin}")
            typer.echo(f"  Transforms: {len(config.row_plugins)}")
            typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
            return

        if not execute:
            typer.echo("Pipeline configuration valid.")
            typer.echo(f"  Source: {config.datasource.plugin}")
            typer.echo("")
            typer.echo("To execute, add --execute flag:", err=True)
            typer.echo(f"  elspeth run -s {settings} --execute", err=True)
            raise typer.Exit(1)
    else:
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
        if output_format == "json":
            import json
            typer.echo(
                json.dumps({
                    "event": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }),
                err=True,
            )
        else:
            typer.echo(f"Error during pipeline execution: {e}", err=True)
        raise typer.Exit(1) from None
```

### Step 4: Implement _execute_pipeline_with_instances (COMPLETE)

**File:** `src/elspeth/cli.py` (add after existing _execute_pipeline)

```python
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
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseSink, BaseTransform
    from elspeth.core.events import EventBus
    import json

    # Use pre-instantiated plugins
    source = plugins["source"]
    sinks = plugins["sinks"]

    # Build transforms list: row_plugins + aggregations (with node_id)
    transforms: list[BaseTransform] = list(plugins["transforms"])

    # Add aggregation transforms with node_id attached
    agg_id_map = graph.get_aggregation_id_map()
    aggregation_settings: dict[str, AggregationSettings] = {}

    for agg_name, (transform, agg_config) in plugins["aggregations"].items():
        node_id = agg_id_map[agg_name]
        aggregation_settings[node_id] = agg_config

        # Set node_id so processor can identify as aggregation
        transform.node_id = node_id
        transforms.append(transform)  # type: ignore[arg-type]

    # Get database
    db_url = config.landscape.url
    db = LandscapeDB.from_url(db_url)

    try:
        # Build PipelineConfig with pre-instantiated plugins
        pipeline_config = PipelineConfig(
            source=source,  # type: ignore[arg-type]
            transforms=transforms,  # type: ignore[arg-type]
            sinks=sinks,  # type: ignore[arg-type]
            config=resolve_config(config),
            gates=list(config.gates),
            aggregation_settings=aggregation_settings,
        )

        if verbose:
            typer.echo("Starting pipeline execution...")

        # Create event bus
        event_bus = EventBus()

        # === EVENT FORMATTER REGISTRATION ===
        # This is the "copy from existing" section - now COMPLETE

        if output_format == "json":
            # JSON mode - structured events
            def handle_row_processed(event):
                typer.echo(json.dumps({
                    "event": "row_processed",
                    "row_id": event.row_id,
                    "token_id": event.token_id,
                    "node_id": event.node_id,
                }))

            def handle_row_quarantined(event):
                typer.echo(json.dumps({
                    "event": "row_quarantined",
                    "row_id": event.row_id,
                    "reason": event.reason,
                }), err=True)

            def handle_row_routed(event):
                typer.echo(json.dumps({
                    "event": "row_routed",
                    "row_id": event.row_id,
                    "sink_name": event.sink_name,
                    "route_label": event.route_label,
                }))

            def handle_batch_emitted(event):
                typer.echo(json.dumps({
                    "event": "batch_emitted",
                    "node_id": event.node_id,
                    "token_id": event.token_id,
                    "batch_size": event.batch_size,
                }))

            def handle_run_started(event):
                typer.echo(json.dumps({
                    "event": "run_started",
                    "run_id": event.run_id,
                }))

            def handle_run_completed(event):
                typer.echo(json.dumps({
                    "event": "run_completed",
                    "run_id": event.run_id,
                    "rows_processed": event.rows_processed,
                    "rows_quarantined": event.rows_quarantined,
                    "status": event.status,
                }))

            # Subscribe JSON handlers
            event_bus.subscribe("row_processed", handle_row_processed)
            event_bus.subscribe("row_quarantined", handle_row_quarantined)
            event_bus.subscribe("row_routed", handle_row_routed)
            event_bus.subscribe("batch_emitted", handle_batch_emitted)
            event_bus.subscribe("run_started", handle_run_started)
            event_bus.subscribe("run_completed", handle_run_completed)

        else:
            # Console mode - human-readable
            from rich.console import Console
            from rich.progress import Progress

            console = Console()
            progress_bar = None

            def handle_run_started(event):
                nonlocal progress_bar
                console.print(f"[bold green]Run started:[/] {event.run_id}")
                if verbose:
                    progress_bar = Progress()
                    progress_bar.start()

            def handle_row_processed(event):
                if verbose and progress_bar:
                    progress_bar.console.print(
                        f"  Row {event.row_id} -> Node {event.node_id}"
                    )

            def handle_row_quarantined(event):
                console.print(
                    f"[yellow]Row quarantined:[/] {event.row_id} - {event.reason}",
                    style="yellow"
                )

            def handle_run_completed(event):
                if progress_bar:
                    progress_bar.stop()
                console.print(f"\n[bold green]Run completed:[/] {event.run_id}")
                console.print(f"  Rows processed: {event.rows_processed}")
                console.print(f"  Rows quarantined: {event.rows_quarantined}")
                console.print(f"  Status: {event.status}")

            # Subscribe console handlers
            event_bus.subscribe("run_started", handle_run_started)
            event_bus.subscribe("row_processed", handle_row_processed)
            event_bus.subscribe("row_quarantined", handle_row_quarantined)
            event_bus.subscribe("run_completed", handle_run_completed)

        # Create and run orchestrator
        orchestrator = Orchestrator(
            pipeline_config=pipeline_config,
            landscape_db=db,
            event_bus=event_bus,
            graph=graph,
        )

        result = orchestrator.run()
        return result

    finally:
        db.close()
```

### Step 5: Run test to verify it passes

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_run_detects_schema_incompatibility -v`

Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/cli.py tests/integration/test_cli_schema_validation.py
git commit -m "refactor(cli): use from_plugin_instances in run command

- Instantiate plugins before graph construction
- Build graph from instances using from_plugin_instances()
- Schema validation now functional
- Add _execute_pipeline_with_instances() with COMPLETE implementation
- Reuse pre-instantiated plugins (no double instantiation)
- Full event formatter registration included
- Add integration test verifying CLI detects schema incompatibility

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 6: Refactor `validate` Command

**Files:**
- Modify: `src/elspeth/cli.py` (validate command)
- Test: `tests/integration/test_cli_schema_validation.py`

**Purpose:** Use new graph construction in validate command.

### Step 1: Write failing test

**File:** `tests/integration/test_cli_schema_validation.py`

```python
def test_cli_validate_detects_schema_incompatibility():
    """Verify CLI validate command detects schema incompatibility."""
    runner = CliRunner()

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        field_a: {type: str}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          field_a: {type: str}

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          field_b: {type: int}  # INCOMPATIBLE

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        result = runner.invoke(app, ["validate", "--settings", str(config_file)])

        # Should fail with schema error
        assert result.exit_code != 0
        assert "schema" in result.output.lower() or "field_b" in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_validate_detects_schema_incompatibility -v`

Expected: FAIL (validate still uses old construction)

### Step 3: Refactor validate command

**File:** `src/elspeth/cli.py` (modify validate command, ~lines 580-635)

```python
@app.command()
def validate(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
) -> None:
    """Validate pipeline configuration without running."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    settings_path = Path(settings).expanduser()

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
            output_sink=config.output_sink,
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo("✅ Pipeline configuration valid!")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Aggregations: {len(config.aggregations)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_cli_schema_validation.py::test_cli_validate_detects_schema_incompatibility -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/integration/test_cli_schema_validation.py
git commit -m "refactor(cli): use from_plugin_instances in validate command

- Instantiate plugins before graph construction
- Schema validation now functional in validate command
- Add integration test verifying validate detects incompatibility

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 7: Update `resume` Command (CRITICAL - Was Missing in v1)

**Files:**
- Modify: `src/elspeth/cli.py` (resume command, ~lines 850-970)
- Test: `tests/integration/test_cli_resume.py` (new test file)

**Purpose:** Update resume command to use new graph construction. **Critical fix** identified by multi-agent review.

### Step 1: Write failing integration test

**File:** `tests/integration/test_cli_resume.py`

```python
"""Integration tests for resume command with new schema validation."""

import tempfile
from pathlib import Path
from typer.testing import CliRunner
from elspeth.cli import app
import pytest


def test_resume_command_uses_new_graph_construction():
    """Verify resume command builds graph from plugin instances."""
    # This test verifies resume doesn't call deprecated from_config()
    # Actual checkpoint/resume testing requires database setup

    runner = CliRunner()

    # Create minimal valid config
    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output

landscape:
  url: "sqlite:///:memory:"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        # Resume with non-existent run_id should fail gracefully
        # but NOT crash due to from_config() deprecation warning
        result = runner.invoke(app, [
            "resume",
            "--settings", str(config_file),
            "--run-id", "nonexistent-run-id",
        ])

        # Should exit with error (run not found), not crash
        assert result.exit_code != 0
        # Should NOT contain deprecation warning
        assert "deprecated" not in result.output.lower()

    finally:
        config_file.unlink()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_resume.py::test_resume_command_uses_new_graph_construction -v`

Expected: FAIL (resume uses from_config, gets deprecation warning)

### Step 3: Update resume command

**File:** `src/elspeth/cli.py` (modify resume command)

```python
@app.command()
def resume(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
    run_id: str = typer.Option(..., "--run-id", help="Run ID to resume."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output."),
    output_format: Literal["console", "json"] = typer.Option("console", "--format", "-f"),
) -> None:
    """Resume a failed or interrupted pipeline run."""
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.dag import ExecutionGraph, GraphValidationError
    from elspeth.core.landscape import LandscapeDB

    settings_path = Path(settings).expanduser()

    # Load config
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1) from None

    # Connect to landscape database
    db_url = config.landscape.url
    db = LandscapeDB.from_url(db_url)

    try:
        # Verify run exists
        run_info = db.get_run_info(run_id)
        if run_info is None:
            typer.echo(f"Error: Run {run_id} not found in database", err=True)
            raise typer.Exit(1) from None

        if verbose:
            typer.echo(f"Resuming run: {run_id}")
            typer.echo(f"  Original status: {run_info.status}")
            typer.echo(f"  Rows processed: {run_info.rows_processed}")

        # NEW: Instantiate plugins BEFORE graph construction
        try:
            plugins = instantiate_plugins_from_config(config)
        except Exception as e:
            typer.echo(f"Error instantiating plugins: {e}", err=True)
            raise typer.Exit(1) from None

        # NEW: Build graph from plugin instances
        # Note: Resume uses NullSource instead of configured source
        from elspeth.plugins.sources.null_source import NullSource

        null_source = NullSource({})
        resume_plugins = {
            **plugins,
            "source": null_source,  # Override with NullSource for resume
        }

        try:
            graph = ExecutionGraph.from_plugin_instances(
                source=resume_plugins["source"],
                transforms=resume_plugins["transforms"],
                sinks=resume_plugins["sinks"],
                aggregations=resume_plugins["aggregations"],
                gates=list(config.gates),
                output_sink=config.output_sink,
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
            graph.validate()
        except GraphValidationError as e:
            typer.echo(f"Pipeline graph error: {e}", err=True)
            raise typer.Exit(1) from None

        # Execute resume with pre-instantiated plugins
        try:
            _execute_resume_with_instances(
                config,
                graph,
                resume_plugins,
                run_id=run_id,
                db=db,
                verbose=verbose,
                output_format=output_format,
            )
        except Exception as e:
            if output_format == "json":
                import json
                typer.echo(
                    json.dumps({
                        "event": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }),
                    err=True,
                )
            else:
                typer.echo(f"Error during resume: {e}", err=True)
            raise typer.Exit(1) from None

    finally:
        db.close()


def _execute_resume_with_instances(
    config: ElspethSettings,
    graph: ExecutionGraph,
    plugins: dict[str, Any],
    run_id: str,
    db: LandscapeDB,
    verbose: bool = False,
    output_format: Literal["console", "json"] = "console",
) -> ExecutionResult:
    """Execute resume using pre-instantiated plugins.

    Similar to _execute_pipeline_with_instances but for resume operations.

    Args:
        config: Validated ElspethSettings
        graph: Validated ExecutionGraph
        plugins: Pre-instantiated plugins (with NullSource)
        run_id: Run ID to resume
        db: LandscapeDB connection
        verbose: Show detailed output
        output_format: 'console' or 'json'

    Returns:
        ExecutionResult
    """
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.core.events import EventBus

    # Build pipeline config (similar to _execute_pipeline_with_instances)
    source = plugins["source"]
    sinks = plugins["sinks"]
    transforms = list(plugins["transforms"])

    agg_id_map = graph.get_aggregation_id_map()
    aggregation_settings = {}

    for agg_name, (transform, agg_config) in plugins["aggregations"].items():
        node_id = agg_id_map[agg_name]
        aggregation_settings[node_id] = agg_config
        transform.node_id = node_id
        transforms.append(transform)

    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,
        sinks=sinks,
        config=resolve_config(config),
        gates=list(config.gates),
        aggregation_settings=aggregation_settings,
    )

    # Create event bus (event handlers same as run command)
    event_bus = EventBus()

    # Create orchestrator for resume
    orchestrator = Orchestrator(
        pipeline_config=pipeline_config,
        landscape_db=db,
        event_bus=event_bus,
        graph=graph,
        resume_run_id=run_id,  # CRITICAL: Pass run_id for resume
    )

    result = orchestrator.resume(run_id)
    return result
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_cli_resume.py::test_resume_command_uses_new_graph_construction -v`

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/integration/test_cli_resume.py
git commit -m "refactor(cli): update resume command to use from_plugin_instances

- CRITICAL FIX: Resume now instantiates plugins before graph construction
- Uses NullSource override for resume operations
- Add _execute_resume_with_instances() helper
- No deprecation warnings (no from_config() usage)
- Add integration test verifying new construction
- Ensures resume works after checkpoint with new architecture

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

**CLI Refactor Complete! Next:** `03-testing.md` for comprehensive test coverage
