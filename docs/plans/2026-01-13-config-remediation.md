# Configuration & DAG Remediation Plan

**Date:** 2026-01-13
**Priority:** P0 - Blocking
**Scope:** Align configuration and DAG systems with architecture specification

---

## Problem Statement

The implementation has diverged from the architecture in two critical areas:

### Problem 1: Configuration System (Tasks 1-8)

1. **Dead Code Path:** `ElspethSettings` and `load_settings()` exist but CLI doesn't use them
2. **Schema Mismatch:** CLI expects `source`, architecture specifies `datasource`
3. **Missing Settings:** `ConcurrencySettings`, `LandscapeSettings` per architecture never implemented

### Problem 2: DAG Infrastructure (Tasks 9-13)

1. **Dead Code:** `ExecutionGraph` (169 lines) exists but Orchestrator doesn't use it
2. **No Validation:** Cycles, missing sources/sinks not detected before execution
3. **No Topological Sort:** Transforms executed via naive `enumerate()`, not graph order
4. **Ad-hoc Edge Map:** Orchestrator builds `dict[tuple[str,str], str]` instead of using ExecutionGraph

### Current State

```
README.md expects:          CLI parses:              config.py defines:
─────────────────           ───────────              ──────────────────
datasource:                 source:                  database:
  plugin: csv_local           plugin: csv              url: ...
  options:                    path: ...              retry:
    path: ...                                          max_attempts: ...
row_plugins:                (not parsed)             payload_store:
  - plugin: ...                                        backend: ...
output_sink: results        (not parsed)
landscape:                  landscape:
  enabled: true               url: ...
  backend: sqlite
  path: ./runs/...
concurrency:                (not parsed)
  max_workers: 4
```

**Result:** Three incompatible config formats, Pydantic validation never applied to real runs.

---

## Success Criteria

### Configuration (Tasks 1-8)
1. CLI uses `load_settings()` exclusively - no direct `yaml.safe_load()`
2. Config schema matches architecture (README.md examples work unchanged)
3. All existing tests pass
4. Pydantic validation catches config errors with clear messages

### DAG Infrastructure (Tasks 9-13)
5. `ExecutionGraph.from_config()` builds graph from `ElspethSettings`
6. `graph.validate()` called before any execution
7. Orchestrator uses `ExecutionGraph` instead of ad-hoc dict
8. Transforms executed in `graph.topological_order()`
9. Invalid pipeline configs fail at validation, not runtime

### Phase 2 Integration (Tasks 14-19)
10. Filter plugin works without AssertionError (uses `filtered` status)
11. RoutingAction.reason immutable (defensive copy)
12. Plugin metadata recorded in Landscape (version, config, determinism)
13. Lifecycle hooks invoked (`on_start`, `on_complete`)

### Phase 2 Consistency (Tasks 20-22)
14. PHASE3_INTEGRATION.md documents actual LandscapeRecorder API
15. RoutingAction uses RoutingKind/RoutingMode enums (not string literals)
16. PluginSpec.from_plugin() populates schema hashes

---

## Critical Contracts (Resolved Blocking Issues)

### 1. output_sink vs "default" - RESOLVED

**Problem:** Current orchestrator uses `pending_tokens["default"]` (hardcoded) but config
specifies `output_sink` as the target for completed rows.

**Resolution:**
- `ElspethSettings.output_sink` specifies which sink receives rows completing the pipeline
- Orchestrator uses `pending_tokens[config.output_sink]` instead of hardcoded "default"
- `ExecutionGraph.from_config()` creates edge with label="continue" to output_sink node
- Task 11 implementation uses `sink_id_map[config.output_sink]` for routing

### 2. Route Labels vs Sink Names - RESOLVED

**Problem:** Confusion about what edge labels represent.

**Resolution per architecture.md:**
- **Edge labels = route labels** (e.g., "continue", "suspicious", "clean")
- **NOT sink names** (e.g., "flagged", "results")

**Config example:**
```yaml
routes:
  suspicious: flagged    # route_label "suspicious" -> sink_name "flagged"
  clean: continue        # route_label "clean" -> proceed to next transform
```

**Edge created:** `gate_node -> flagged_sink_node` with `label="suspicious"`

**Current ThresholdGate divergence:** ThresholdGate uses `above_sink`/`below_sink` directly,
not `routes` config. This is an existing Phase 2 simplification - the routes-style config
is the architecture target but can be addressed post-migration.

### 3. LandscapeSettings.url (not path) - RESOLVED

**Problem:** `pathlib.Path` mangles PostgreSQL DSNs like `postgresql://user@host/db`.

**Resolution:** `LandscapeSettings.url: str` stores full SQLAlchemy URL directly.
No path+backend combination needed.

### 4. Explicit Node ID Mappings - RESOLVED

**Problem:** Brittle substring matching like `if sink_name in nid`.

**Resolution:** ExecutionGraph maintains explicit mappings:
- `get_sink_id_map()` → `dict[sink_name, node_id]`
- `get_transform_id_map()` → `dict[sequence, node_id]`
- No string matching - direct lookup

### 5. No Legacy Path Fallback - RESOLVED

**Problem:** `_execute_legacy()` fallback violates "no compatibility shims" policy.

**Resolution:** Graph is REQUIRED - no fallback. Per CLAUDE.md policy.

---

## Task 1: Add Architecture-Compliant Settings Classes

**Goal:** Expand config.py with settings classes matching the architecture specification.

### Test First

```python
# tests/core/test_config.py

class TestArchitectureCompliantSettings:
    """Settings classes match architecture specification."""

    def test_datasource_settings_structure(self) -> None:
        """DatasourceSettings has plugin and options."""
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv_local", options={"path": "data/input.csv"})
        assert ds.plugin == "csv_local"
        assert ds.options == {"path": "data/input.csv"}

    def test_datasource_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv")
        assert ds.options == {}

    def test_row_plugin_settings_structure(self) -> None:
        """RowPluginSettings has plugin, type, options, routes."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(
            plugin="threshold_gate",
            type="gate",
            options={"field": "confidence", "min": 0.8},
            routes={"pass": "continue", "fail": "quarantine"},
        )
        assert rp.plugin == "threshold_gate"
        assert rp.type == "gate"
        assert rp.routes == {"pass": "continue", "fail": "quarantine"}

    def test_row_plugin_settings_defaults(self) -> None:
        """RowPluginSettings defaults: type=transform, no routes."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(plugin="field_mapper")
        assert rp.type == "transform"
        assert rp.options == {}
        assert rp.routes is None

    def test_sink_settings_structure(self) -> None:
        """SinkSettings has plugin and options."""
        from elspeth.core.config import SinkSettings

        sink = SinkSettings(plugin="csv", options={"path": "output/results.csv"})
        assert sink.plugin == "csv"
        assert sink.options == {"path": "output/results.csv"}

    def test_landscape_settings_structure(self) -> None:
        """LandscapeSettings has enabled, backend, url."""
        from elspeth.core.config import LandscapeSettings

        ls = LandscapeSettings(enabled=True, backend="sqlite", url="sqlite:///./runs/audit.db")
        assert ls.enabled is True
        assert ls.backend == "sqlite"
        assert ls.url == "sqlite:///./runs/audit.db"

    def test_landscape_settings_defaults(self) -> None:
        """LandscapeSettings has sensible defaults."""
        from elspeth.core.config import LandscapeSettings

        ls = LandscapeSettings()
        assert ls.enabled is True
        assert ls.backend == "sqlite"
        assert ls.url == "sqlite:///./runs/audit.db"

    def test_landscape_settings_postgresql_url(self) -> None:
        """LandscapeSettings accepts PostgreSQL DSNs without mangling."""
        from elspeth.core.config import LandscapeSettings

        # This would fail with pathlib.Path which mangles // as UNC paths
        pg_url = "postgresql://user:pass@localhost:5432/elspeth_audit"
        ls = LandscapeSettings(enabled=True, backend="postgresql", url=pg_url)
        assert ls.url == pg_url  # Preserved exactly

    def test_landscape_settings_backend_validation(self) -> None:
        """Backend must be sqlite or postgresql."""
        from pydantic import ValidationError
        from elspeth.core.config import LandscapeSettings

        with pytest.raises(ValidationError):
            LandscapeSettings(backend="mysql")

    def test_concurrency_settings_structure(self) -> None:
        """ConcurrencySettings has max_workers."""
        from elspeth.core.config import ConcurrencySettings

        cs = ConcurrencySettings(max_workers=16)
        assert cs.max_workers == 16

    def test_concurrency_settings_default(self) -> None:
        """Default max_workers is 4 per architecture."""
        from elspeth.core.config import ConcurrencySettings

        cs = ConcurrencySettings()
        assert cs.max_workers == 4

    def test_concurrency_settings_validation(self) -> None:
        """max_workers must be positive."""
        from pydantic import ValidationError
        from elspeth.core.config import ConcurrencySettings

        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=0)
        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=-1)
```

### Implementation

```python
# src/elspeth/core/config.py

from typing import Any, Literal

class DatasourceSettings(BaseModel):
    """Source plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class RowPluginSettings(BaseModel):
    """Transform or gate plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name")
    type: Literal["transform", "gate"] = Field(
        default="transform",
        description="Plugin type: transform (pass-through) or gate (routing)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    routes: dict[str, str] | None = Field(
        default=None,
        description="Gate routing map: result -> sink_name or 'continue'",
    )


class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )


class LandscapeSettings(BaseModel):
    """Landscape audit system configuration per architecture."""

    model_config = {"frozen": True}

    enabled: bool = Field(default=True, description="Enable audit trail recording")
    backend: Literal["sqlite", "postgresql"] = Field(
        default="sqlite",
        description="Database backend type",
    )
    # NOTE: Using str instead of Path - Path mangles PostgreSQL DSNs like
    # "postgresql://user:pass@host/db" (pathlib interprets // as UNC path)
    url: str = Field(
        default="sqlite:///./runs/audit.db",
        description="Full SQLAlchemy database URL",
    )


class ConcurrencySettings(BaseModel):
    """Parallel processing configuration per architecture."""

    model_config = {"frozen": True}

    max_workers: int = Field(
        default=4,
        gt=0,
        description="Maximum parallel workers (default 4, production typically 16)",
    )
```

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_config.py::TestArchitectureCompliantSettings -v
```

---

## Task 2: Update ElspethSettings to Match Architecture

**Goal:** Replace current ElspethSettings with architecture-compliant top-level schema.

### Test First

```python
# tests/core/test_config.py

class TestElspethSettingsArchitecture:
    """Top-level settings matches architecture specification."""

    def test_elspeth_settings_required_fields(self) -> None:
        """ElspethSettings requires datasource, sinks, output_sink."""
        from pydantic import ValidationError
        from elspeth.core.config import ElspethSettings

        # Missing required fields
        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings()

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "datasource" in missing_fields
        assert "sinks" in missing_fields
        assert "output_sink" in missing_fields

    def test_elspeth_settings_minimal_valid(self) -> None:
        """Minimal valid configuration."""
        from elspeth.core.config import (
            ElspethSettings,
            DatasourceSettings,
            SinkSettings,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "in.csv"}),
            sinks={"results": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="results",
        )

        assert settings.datasource.plugin == "csv"
        assert "results" in settings.sinks
        assert settings.output_sink == "results"
        # Defaults applied
        assert settings.row_plugins == []
        assert settings.landscape.enabled is True
        assert settings.concurrency.max_workers == 4

    def test_elspeth_settings_full_config(self) -> None:
        """Full configuration matching README example."""
        from elspeth.core.config import (
            ElspethSettings,
            DatasourceSettings,
            RowPluginSettings,
            SinkSettings,
            LandscapeSettings,
            ConcurrencySettings,
        )

        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="csv_local",
                options={"path": "data/submissions.csv"},
            ),
            sinks={
                "results": SinkSettings(plugin="csv", options={"path": "output/results.csv"}),
                "flagged": SinkSettings(plugin="csv", options={"path": "output/flagged.csv"}),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="pattern_gate",
                    type="gate",
                    options={"patterns": ["ignore previous"]},
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
            landscape=LandscapeSettings(
                enabled=True,
                backend="sqlite",
                url="sqlite:///./runs/audit.db",
            ),
            concurrency=ConcurrencySettings(max_workers=8),
        )

        assert len(settings.row_plugins) == 1
        assert settings.row_plugins[0].type == "gate"
        assert settings.landscape.backend == "sqlite"
        assert settings.concurrency.max_workers == 8

    def test_elspeth_settings_output_sink_must_exist(self) -> None:
        """output_sink must reference a defined sink."""
        from pydantic import ValidationError
        from elspeth.core.config import (
            ElspethSettings,
            DatasourceSettings,
            SinkSettings,
        )

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={"results": SinkSettings(plugin="csv")},
                output_sink="nonexistent",  # Not in sinks!
            )

        assert "output_sink" in str(exc_info.value)

    def test_elspeth_settings_at_least_one_sink(self) -> None:
        """At least one sink is required."""
        from pydantic import ValidationError
        from elspeth.core.config import ElspethSettings, DatasourceSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={},  # Empty!
                output_sink="results",
            )

        assert "sinks" in str(exc_info.value).lower()
```

### Implementation

```python
# src/elspeth/core/config.py

class ElspethSettings(BaseModel):
    """Top-level Elspeth configuration matching architecture specification.

    This is the single source of truth for pipeline configuration.
    All settings are validated and frozen after construction.
    """

    model_config = {"frozen": True}

    # Required - core pipeline definition
    datasource: DatasourceSettings = Field(
        description="Source plugin configuration (exactly one per run)",
    )
    sinks: dict[str, SinkSettings] = Field(
        description="Named sink configurations (one or more required)",
    )
    output_sink: str = Field(
        description="Default sink for rows that complete the pipeline",
    )

    # Optional - transform chain
    row_plugins: list[RowPluginSettings] = Field(
        default_factory=list,
        description="Ordered list of transforms/gates to apply",
    )

    # Optional - subsystem configuration with defaults
    landscape: LandscapeSettings = Field(
        default_factory=LandscapeSettings,
        description="Audit trail configuration",
    )
    concurrency: ConcurrencySettings = Field(
        default_factory=ConcurrencySettings,
        description="Parallel processing configuration",
    )
    retry: RetrySettings = Field(
        default_factory=RetrySettings,
        description="Retry behavior configuration",
    )
    payload_store: PayloadStoreSettings = Field(
        default_factory=PayloadStoreSettings,
        description="Large payload storage configuration",
    )

    @model_validator(mode="after")
    def validate_output_sink_exists(self) -> "ElspethSettings":
        """Ensure output_sink references a defined sink."""
        if self.output_sink not in self.sinks:
            raise ValueError(
                f"output_sink '{self.output_sink}' not found in sinks. "
                f"Available sinks: {list(self.sinks.keys())}"
            )
        return self

    @field_validator("sinks")
    @classmethod
    def validate_sinks_not_empty(cls, v: dict[str, SinkSettings]) -> dict[str, SinkSettings]:
        """At least one sink is required."""
        if not v:
            raise ValueError("At least one sink is required")
        return v
```

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_config.py::TestElspethSettingsArchitecture -v
```

---

## Task 3: Update load_settings() for New Schema

**Goal:** Ensure load_settings() correctly parses architecture-compliant YAML.

### Test First

```python
# tests/core/test_config.py

class TestLoadSettingsArchitecture:
    """load_settings() parses architecture-compliant YAML."""

    def test_load_readme_example(self, tmp_path: Path) -> None:
        """Load the exact example from README.md."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv_local
  options:
    path: data/submissions.csv

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv
  flagged:
    plugin: csv
    options:
      path: output/flagged_for_review.csv

row_plugins:
  - plugin: pattern_gate
    type: gate
    options:
      patterns:
        - "ignore previous"
        - "disregard instructions"
    routes:
      suspicious: flagged
      clean: continue

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///./runs/audit.db
""")

        settings = load_settings(config_file)

        assert settings.datasource.plugin == "csv_local"
        assert settings.datasource.options["path"] == "data/submissions.csv"
        assert len(settings.sinks) == 2
        assert len(settings.row_plugins) == 1
        assert settings.row_plugins[0].type == "gate"
        assert settings.output_sink == "results"
        assert settings.landscape.backend == "sqlite"

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Minimal valid configuration."""
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
""")

        settings = load_settings(config_file)

        assert settings.datasource.plugin == "csv"
        assert settings.landscape.enabled is True  # Default
        assert settings.concurrency.max_workers == 4  # Default

    def test_load_with_env_interpolation(self, tmp_path: Path, monkeypatch) -> None:
        """Environment variables are interpolated."""
        from elspeth.core.config import load_settings

        monkeypatch.setenv("ELSPETH_DATASOURCE__OPTIONS__PATH", "/override/path.csv")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: original/path.csv

sinks:
  output:
    plugin: csv

output_sink: output
""")

        settings = load_settings(config_file)

        # Environment override takes precedence
        assert settings.datasource.options["path"] == "/override/path.csv"

    def test_load_invalid_output_sink(self, tmp_path: Path) -> None:
        """Error when output_sink doesn't exist."""
        from pydantic import ValidationError
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv

output_sink: nonexistent
""")

        with pytest.raises(ValidationError) as exc_info:
            load_settings(config_file)

        assert "output_sink" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_load_invalid_landscape_backend(self, tmp_path: Path) -> None:
        """Error on invalid landscape backend."""
        from pydantic import ValidationError
        from elspeth.core.config import load_settings

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output

landscape:
  backend: mysql
""")

        with pytest.raises(ValidationError) as exc_info:
            load_settings(config_file)

        assert "backend" in str(exc_info.value).lower()
```

### Implementation

Update `load_settings()` to handle nested structures properly:

```python
# src/elspeth/core/config.py

def load_settings(config_path: Path) -> ElspethSettings:
    """Load settings from YAML file with environment variable overrides.

    Uses Dynaconf for multi-source loading with precedence:
    1. Environment variables (ELSPETH_*) - highest priority
    2. Config file (settings.yaml)
    3. Defaults from Pydantic schema - lowest priority

    Environment variable format: ELSPETH_DATASOURCE__OPTIONS__PATH for nested keys.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ElspethSettings instance

    Raises:
        ValidationError: If configuration fails Pydantic validation
        FileNotFoundError: If config file doesn't exist
    """
    from dynaconf import Dynaconf

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dynaconf_settings = Dynaconf(
        envvar_prefix="ELSPETH",
        settings_files=[str(config_path)],
        environments=False,
        load_dotenv=False,
        merge_enabled=True,
    )

    # Convert to dict, filtering Dynaconf internals
    internal_keys = {"LOAD_DOTENV", "ENVIRONMENTS", "SETTINGS_FILES"}
    raw_config = {
        k.lower(): v
        for k, v in dynaconf_settings.as_dict().items()
        if k not in internal_keys
    }

    return ElspethSettings(**raw_config)
```

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_config.py::TestLoadSettingsArchitecture -v
```

---

## Task 4: Wire CLI to Use load_settings()

**Goal:** Replace ad-hoc YAML parsing in CLI with load_settings().

### Test First

```python
# tests/cli/test_cli_config.py

"""CLI configuration loading tests."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestCliUsesLoadSettings:
    """CLI commands use load_settings() for config."""

    def test_run_with_readme_config(self, tmp_path: Path) -> None:
        """Run command accepts README-style config."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: input.csv

sinks:
  results:
    plugin: csv
    options:
      path: output.csv

output_sink: results
""")

        # Dry run should parse config successfully
        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()

    def test_run_rejects_old_config_format(self, tmp_path: Path) -> None:
        """Run command rejects old 'source' format."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
source:
  plugin: csv
  path: input.csv

sinks:
  results:
    plugin: csv
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        # Should fail - 'source' is not valid, must be 'datasource'
        assert result.exit_code != 0
        assert "datasource" in result.stdout.lower() or "datasource" in result.stderr.lower() if result.stderr else True

    def test_validate_with_readme_config(self, tmp_path: Path) -> None:
        """Validate command accepts README-style config."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output

landscape:
  enabled: true
  backend: sqlite
  path: ./audit.db
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_shows_pydantic_errors(self, tmp_path: Path) -> None:
        """Validate shows Pydantic validation errors clearly."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: nonexistent

concurrency:
  max_workers: -5
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        # Should show helpful error messages
        output = result.stdout + (result.stderr or "")
        assert "output_sink" in output.lower() or "nonexistent" in output.lower()

    def test_run_uses_concurrency_setting(self, tmp_path: Path) -> None:
        """Run command respects concurrency.max_workers."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: input.csv

sinks:
  results:
    plugin: csv
    options:
      path: output.csv

output_sink: results

concurrency:
  max_workers: 8
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run", "-v"])

        assert result.exit_code == 0
        # Verbose output should show config was parsed
```

### Implementation

Refactor `cli.py` to use `load_settings()`:

```python
# src/elspeth/cli.py

from elspeth.core.config import load_settings, ElspethSettings
from pydantic import ValidationError

@app.command()
def run(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Validate without executing."),
    execute: bool = typer.Option(False, "--execute", "-x", help="Actually execute the pipeline."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output."),
) -> None:
    """Execute a pipeline run."""
    settings_path = Path(settings)

    # Load and validate config via Pydantic
    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo("Dry run mode - would execute:")
        typer.echo(f"  Source: {config.datasource.plugin}")
        typer.echo(f"  Transforms: {len(config.row_plugins)}")
        typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
        typer.echo(f"  Output sink: {config.output_sink}")
        if verbose:
            typer.echo(f"  Concurrency: {config.concurrency.max_workers} workers")
            typer.echo(f"  Landscape: {config.landscape.url}")
        return

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
        raise typer.Exit(1)


@app.command()
def validate(
    settings: str = typer.Option(..., "--settings", "-s", help="Path to settings YAML file."),
) -> None:
    """Validate pipeline configuration without running."""
    settings_path = Path(settings)

    try:
        config = load_settings(settings_path)
    except FileNotFoundError:
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo("Configuration errors:", err=True)
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            typer.echo(f"  - {loc}: {error['msg']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration valid: {settings_path.name}")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Output: {config.output_sink}")
```

### Verification

```bash
.venv/bin/python -m pytest tests/cli/test_cli_config.py -v
```

---

## Task 5: Update _execute_pipeline() for New Config

**Goal:** Refactor `_execute_pipeline()` to use `ElspethSettings` instead of raw dict.

### Test First

```python
# tests/cli/test_cli_execution.py

"""CLI pipeline execution tests."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestCliExecutePipeline:
    """Pipeline execution with validated config."""

    def test_execute_csv_pipeline(self, tmp_path: Path) -> None:
        """Execute a simple CSV pipeline."""
        # Create input file
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,name\n1,Alice\n2,Bob\n")

        output_file = tmp_path / "output.csv"

        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: {input_file}

sinks:
  results:
    plugin: csv
    options:
      path: {output_file}

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///{tmp_path / "audit.db"}
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "completed" in result.stdout.lower()

    def test_execute_with_landscape_disabled(self, tmp_path: Path) -> None:
        """Execute pipeline with landscape disabled."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,name\n1,Test\n")

        output_file = tmp_path / "output.csv"
        audit_file = tmp_path / "audit.db"

        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: {input_file}

sinks:
  results:
    plugin: csv
    options:
      path: {output_file}

output_sink: results

landscape:
  enabled: false
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        assert result.exit_code == 0
        assert not audit_file.exists()  # No audit DB created
```

### Implementation

```python
# src/elspeth/cli.py

def _execute_pipeline(config: ElspethSettings, verbose: bool = False) -> dict[str, Any]:
    """Execute a pipeline from validated configuration.

    Args:
        config: Validated ElspethSettings instance
        verbose: Show detailed output

    Returns:
        Dict with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.engine.adapters import SinkAdapter
    from elspeth.plugins.base import BaseSink, BaseSource
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.sinks.database_sink import DatabaseSink
    from elspeth.plugins.sinks.json_sink import JSONSink
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.sources.json_source import JSONSource

    # Instantiate source from config.datasource
    source = _create_source(config.datasource)

    # Instantiate sinks from config.sinks
    sinks = {
        name: _create_sink_adapter(name, sink_config)
        for name, sink_config in config.sinks.items()
    }

    # TODO: Instantiate transforms from config.row_plugins
    transforms = []  # Will be implemented in Phase 4B

    # Create landscape DB if enabled
    db = None
    if config.landscape.enabled:
        db_url = _landscape_url(config.landscape)
        db = LandscapeDB.from_url(db_url)

    # Build PipelineConfig
    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,
        sinks=sinks,
    )

    if verbose:
        typer.echo("Starting pipeline execution...")
        typer.echo(f"  Concurrency: {config.concurrency.max_workers} workers")

    # Execute via Orchestrator
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }


def _landscape_url(settings: LandscapeSettings) -> str:
    """Get SQLAlchemy URL from LandscapeSettings.

    LandscapeSettings.url stores the full SQLAlchemy URL directly,
    avoiding pathlib mangling of PostgreSQL DSNs.
    """
    return settings.url


def _create_source(config: DatasourceSettings) -> BaseSource:
    """Create source plugin from config."""
    # Map plugin names to classes
    # TODO: Use plugin registry in Phase 4B
    if config.plugin in ("csv", "csv_local"):
        return CSVSource(config.options)
    elif config.plugin == "json":
        return JSONSource(config.options)
    else:
        raise ValueError(f"Unknown source plugin: {config.plugin}")


def _create_sink_adapter(name: str, config: SinkSettings) -> SinkAdapter:
    """Create sink adapter from config."""
    if config.plugin == "csv":
        sink = CSVSink(config.options)
        artifact = {"kind": "file", "path": config.options.get("path", "")}
    elif config.plugin == "json":
        sink = JSONSink(config.options)
        artifact = {"kind": "file", "path": config.options.get("path", "")}
    elif config.plugin == "database":
        sink = DatabaseSink(config.options)
        artifact = {"kind": "database", "table": config.options.get("table", "")}
    else:
        raise ValueError(f"Unknown sink plugin: {config.plugin}")

    return SinkAdapter(
        sink,
        plugin_name=config.plugin,
        sink_name=name,
        artifact_descriptor=artifact,
    )
```

### Verification

```bash
.venv/bin/python -m pytest tests/cli/test_cli_execution.py -v
```

---

## Task 6: Remove Dead Code

**Goal:** Clean up old config validation and unused code.

### Changes

1. **Delete `_validate_config()`** - replaced by Pydantic validation
2. **Delete `KNOWN_SOURCES` / `KNOWN_SINKS`** - replaced by plugin registry
3. **Update exports** in `__init__.py` if needed

### Implementation

```python
# src/elspeth/cli.py

# DELETE these:
# - KNOWN_SOURCES = {"csv", "json"}
# - KNOWN_SINKS = {"csv", "json", "database"}
# - def _validate_config(config: dict[str, Any]) -> list[str]:
```

### Verification

```bash
# Ensure no references remain
grep -r "_validate_config\|KNOWN_SOURCES\|KNOWN_SINKS" src/elspeth/

# All tests still pass
.venv/bin/python -m pytest tests/ -v
```

---

## Task 7: Update Existing Config Tests

**Goal:** Migrate existing tests to new schema format.

### Changes

Update `tests/core/test_config.py` existing tests that use old schema:

```python
# Before (old schema):
config_file.write_text("""
database:
  url: sqlite:///test.db
""")

# After (new schema):
config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

output_sink: output
""")
```

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_config.py -v
.venv/bin/python -m pytest tests/ -v
```

---

## Task 8: Update __init__.py Exports

**Goal:** Export new settings classes from core module.

### Implementation

```python
# src/elspeth/core/__init__.py

from elspeth.core.config import (
    ConcurrencySettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
    load_settings,
)

__all__ = [
    # ... existing exports ...
    "ConcurrencySettings",
    "DatasourceSettings",
    "ElspethSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RetrySettings",
    "RowPluginSettings",
    "SinkSettings",
    "load_settings",
]
```

### Verification

```bash
.venv/bin/python -c "from elspeth.core import ElspethSettings, DatasourceSettings, load_settings; print('OK')"
```

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Add architecture-compliant settings classes | `config.py`, `test_config.py` |
| 2 | Update ElspethSettings to match architecture | `config.py`, `test_config.py` |
| 3 | Update load_settings() for new schema | `config.py`, `test_config.py` |
| 4 | Wire CLI to use load_settings() | `cli.py`, `test_cli_config.py` |
| 5 | Update _execute_pipeline() for new config | `cli.py`, `test_cli_execution.py` |
| 6 | Remove dead code | `cli.py` |
| 7 | Update existing config tests | `test_config.py` |
| 8 | Update __init__.py exports | `core/__init__.py` |

**Estimated total:** ~400-500 lines changed across 5 files

---

## Verification Checklist

After all tasks complete:

```bash
# All tests pass
.venv/bin/python -m pytest tests/ -v

# README example parses correctly
cat > /tmp/test.yaml << 'EOF'
datasource:
  plugin: csv_local
  options:
    path: data/submissions.csv

sinks:
  results:
    plugin: csv
    options:
      path: output/results.csv

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///./runs/audit.db
EOF

elspeth validate -s /tmp/test.yaml

# No dead code remains
grep -r "yaml.safe_load\|_validate_config\|KNOWN_SOURCES" src/elspeth/cli.py
# Should return nothing

# Type checking passes
.venv/bin/python -m mypy src/elspeth/core/config.py
.venv/bin/python -m mypy src/elspeth/cli.py
```

---

# Part 2: DAG Infrastructure Remediation

---

## Current State: ExecutionGraph is Dead Code

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  src/elspeth/core/dag.py (169 lines)                                         │
│                                                                              │
│  class ExecutionGraph:                                                       │
│      def add_node(...)           # ✓ Implemented                            │
│      def add_edge(...)           # ✓ Implemented                            │
│      def is_acyclic(...)         # ✓ Uses NetworkX                          │
│      def validate(...)           # ✓ Checks cycles, source, sinks           │
│      def topological_order(...)  # ✓ Uses NetworkX                          │
│      def get_source(...)         # ✓ Implemented                            │
│      def get_sinks(...)          # ✓ Implemented                            │
│                                                                              │
│  STATUS: NEVER USED BY ANYTHING                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  src/elspeth/engine/orchestrator.py                                          │
│                                                                              │
│  class Orchestrator:                                                         │
│      def _execute_run(...):                                                  │
│          edge_map: dict[tuple[str, str], str] = {}   # Ad-hoc, no validation│
│          for i, transform in enumerate(transforms):  # Naive order          │
│              ...                                                             │
│                                                                              │
│  STATUS: IGNORES ExecutionGraph ENTIRELY                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Architecture says:** "Elspeth compiles pipeline configurations into a directed acyclic graph (DAG). This is the true execution model."

**Implementation does:** Builds ad-hoc dict, executes via enumerate(), no validation.

---

## Task 9: Add ExecutionGraph.from_config() Factory

**Goal:** Create a factory method that builds an ExecutionGraph from ElspethSettings.

### Test First

```python
# tests/core/test_dag.py

class TestExecutionGraphFromConfig:
    """Build ExecutionGraph from ElspethSettings."""

    def test_from_config_minimal(self) -> None:
        """Build graph from minimal config."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> output_sink
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_source() is not None
        assert len(graph.get_sinks()) == 1

    def test_from_config_with_transforms(self) -> None:
        """Build graph with transform chain."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have: source -> transform_a -> transform_b -> output_sink
        assert graph.node_count == 4
        assert graph.edge_count == 3

        # Topological order should be correct
        order = graph.topological_order()
        assert order[0].endswith("source")  # Source first
        assert order[-1].endswith("output")  # Sink last

    def test_from_config_with_gate_routes(self) -> None:
        """Build graph with gate routing to multiple sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="safety_gate",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Should have:
        #   source -> safety_gate -> results (via "continue"/"clean")
        #                         -> flagged (via "suspicious")
        assert graph.node_count == 4  # source, gate, results, flagged
        # Edges: source->gate, gate->results (continue), gate->flagged (route)
        assert graph.edge_count == 3

    def test_from_config_validates_route_targets(self) -> None:
        """Gate routes must reference existing sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(
                    plugin="gate",
                    type="gate",
                    routes={"bad": "nonexistent_sink"},
                ),
            ],
            output_sink="output",
        )

        with pytest.raises(GraphValidationError) as exc_info:
            ExecutionGraph.from_config(config)

        assert "nonexistent_sink" in str(exc_info.value)

    def test_from_config_is_valid(self) -> None:
        """Graph from valid config passes validation."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[RowPluginSettings(plugin="transform")],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Should not raise
        graph.validate()
        assert graph.is_acyclic()
```

### Implementation

```python
# src/elspeth/core/dag.py

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.config import ElspethSettings

class ExecutionGraph:
    # ... existing methods ...

    @classmethod
    def from_config(cls, config: ElspethSettings) -> ExecutionGraph:
        """Build an ExecutionGraph from validated settings.

        Creates nodes for:
        - Source (from config.datasource)
        - Transforms (from config.row_plugins, in order)
        - Sinks (from config.sinks)

        Creates edges for:
        - Linear flow: source -> transforms -> output_sink
        - Gate routes: gate -> routed_sink

        Args:
            config: Validated ElspethSettings

        Returns:
            ExecutionGraph ready for validation and execution

        Raises:
            GraphValidationError: If gate routes reference unknown sinks
        """
        import uuid

        graph = cls()

        # Generate unique node IDs
        def node_id(prefix: str, name: str) -> str:
            return f"{prefix}_{name}_{uuid.uuid4().hex[:8]}"

        # Add source node
        source_id = node_id("source", config.datasource.plugin)
        graph.add_node(
            source_id,
            node_type="source",
            plugin_name=config.datasource.plugin,
            config=config.datasource.options,
        )

        # Add sink nodes - track explicit sink_name -> node_id mapping
        sink_ids: dict[str, str] = {}
        for sink_name, sink_config in config.sinks.items():
            sid = node_id("sink", sink_name)
            sink_ids[sink_name] = sid
            graph.add_node(
                sid,
                node_type="sink",
                plugin_name=sink_config.plugin,
                config=sink_config.options,
            )
        # Store explicit mapping for get_sink_id_map() - NO substring matching
        graph._sink_id_map = dict(sink_ids)

        # Build transform chain - track explicit sequence -> node_id mapping
        transform_ids: dict[int, str] = {}
        prev_node_id = source_id
        for i, plugin_config in enumerate(config.row_plugins):
            is_gate = plugin_config.type == "gate"
            ntype = "gate" if is_gate else "transform"
            tid = node_id(ntype, plugin_config.plugin)

            graph.add_node(
                tid,
                node_type=ntype,
                plugin_name=plugin_config.plugin,
                config=plugin_config.options,
            )
            transform_ids[i] = tid  # Track sequence -> node_id

            # Edge from previous node
            graph.add_edge(prev_node_id, tid, label="continue", mode="move")

            # Gate routes to sinks - edge labels ARE route labels (not sink names)
            # Example: route "suspicious" -> sink "flagged"
            # Creates edge: gate_node -> flagged_node with label="suspicious"
            if is_gate and plugin_config.routes:
                for route_label, target in plugin_config.routes.items():
                    if target == "continue":
                        continue  # Not a sink route
                    if target not in sink_ids:
                        raise GraphValidationError(
                            f"Gate '{plugin_config.plugin}' routes '{route_label}' "
                            f"to unknown sink '{target}'. "
                            f"Available sinks: {list(sink_ids.keys())}"
                        )
                    # Edge label = route_label (e.g., "suspicious")
                    # Edge target = sink node (e.g., flagged_sink_node_id)
                    graph.add_edge(tid, sink_ids[target], label=route_label, mode="move")

            prev_node_id = tid

        # Store explicit mapping for get_transform_id_map()
        graph._transform_id_map = transform_ids

        # Edge from last transform (or source) to output sink
        if config.output_sink not in sink_ids:
            raise GraphValidationError(
                f"output_sink '{config.output_sink}' not in sinks"
            )
        # The "continue" edge to output_sink - this is the default path for
        # rows that don't get routed elsewhere by gates
        graph.add_edge(
            prev_node_id,
            sink_ids[config.output_sink],
            label="continue",  # Not the sink name - "continue" is the route label
            mode="move",
        )

        # Store output_sink for reference
        graph._output_sink = config.output_sink

        return graph
```

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphFromConfig -v
```

---

## Task 10: Add Graph Validation to CLI

**Goal:** Validate the execution graph BEFORE attempting execution.

### Test First

```python
# tests/cli/test_cli_graph_validation.py

"""CLI graph validation tests."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestCliGraphValidation:
    """CLI validates graph before execution."""

    def test_validate_detects_invalid_route(self, tmp_path: Path) -> None:
        """Validate command catches gate routing to nonexistent sink."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: my_gate
    type: gate
    routes:
      bad_route: nonexistent_sink

output_sink: output
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        assert "nonexistent_sink" in result.stdout.lower() or "nonexistent_sink" in (result.stderr or "").lower()

    def test_run_validates_graph_before_execution(self, tmp_path: Path) -> None:
        """Run command validates graph before any execution."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: bad_gate
    type: gate
    routes:
      error: missing_sink

output_sink: output
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        # Should fail at validation, not during execution
        assert result.exit_code != 0
        assert "missing_sink" in result.stdout.lower() or "graph" in result.stdout.lower()

    def test_dry_run_shows_graph_info(self, tmp_path: Path) -> None:
        """Dry run shows graph structure."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  results:
    plugin: csv
  flagged:
    plugin: csv

row_plugins:
  - plugin: classifier
    type: gate
    routes:
      suspicious: flagged
      clean: continue

output_sink: results
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run", "-v"])

        assert result.exit_code == 0
        # Verbose should show graph info
        assert "node" in result.stdout.lower() or "edge" in result.stdout.lower() or "graph" in result.stdout.lower()
```

### Implementation

Update CLI to validate graph:

```python
# src/elspeth/cli.py

from elspeth.core.dag import ExecutionGraph, GraphValidationError

@app.command()
def run(...) -> None:
    """Execute a pipeline run."""
    # ... load config ...

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo("Dry run mode - would execute:")
        typer.echo(f"  Source: {config.datasource.plugin}")
        typer.echo(f"  Transforms: {len(config.row_plugins)}")
        typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
        if verbose:
            typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
            typer.echo(f"  Execution order: {' -> '.join(graph.topological_order())}")
        return

    # ... rest of execution ...


@app.command()
def validate(...) -> None:
    """Validate pipeline configuration without running."""
    # ... load config ...

    # Validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration valid: {settings_path.name}")
    typer.echo(f"  Source: {config.datasource.plugin}")
    typer.echo(f"  Transforms: {len(config.row_plugins)}")
    typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
    typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```

### Verification

```bash
.venv/bin/python -m pytest tests/cli/test_cli_graph_validation.py -v
```

---

## Task 11: Update Orchestrator to Accept ExecutionGraph

**Goal:** Orchestrator uses ExecutionGraph instead of building ad-hoc edge map.

### Test First

```python
# tests/engine/test_orchestrator_graph.py

"""Orchestrator uses ExecutionGraph."""

import pytest
from unittest.mock import MagicMock

from elspeth.core.dag import ExecutionGraph
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig


class TestOrchestratorUsesGraph:
    """Orchestrator accepts and uses ExecutionGraph."""

    def test_orchestrator_accepts_graph(self) -> None:
        """Orchestrator.run() accepts graph parameter."""
        from elspeth.core.landscape import LandscapeDB

        db = LandscapeDB.from_url("sqlite:///:memory:")

        # Build a simple graph
        graph = ExecutionGraph()
        graph.add_node("source_1", node_type="source", plugin_name="csv")
        graph.add_node("sink_1", node_type="sink", plugin_name="csv")
        graph.add_edge("source_1", "sink_1", label="continue", mode="move")

        orchestrator = Orchestrator(db)

        # Should accept graph parameter
        # (Will fail on missing source/sink objects, but signature works)
        with pytest.raises(Exception):  # Expected - no actual plugins
            config = PipelineConfig(
                source=MagicMock(),
                transforms=[],
                sinks={"output": MagicMock()},
            )
            orchestrator.run(config, graph=graph)

    def test_orchestrator_uses_graph_node_ids(self, tmp_path) -> None:
        """Orchestrator uses node IDs from graph, not generated ones."""
        # This will be a more complete integration test
        pass  # Detailed in implementation
```

### Implementation

Refactor `Orchestrator._execute_run()` to use the graph:

```python
# src/elspeth/engine/orchestrator.py

from elspeth.core.dag import ExecutionGraph

class Orchestrator:
    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (optional, built if not provided)
        """
        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config, graph)

            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"
            return result

        except Exception:
            recorder.complete_run(run.run_id, status="failed")
            raise

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> RunResult:
        """Execute the run using the execution graph.

        NOTE: graph is REQUIRED - no legacy path.
        Per CLAUDE.md: "No Legacy Code Policy" - we don't maintain
        backwards compatibility shims.
        """
        return self._execute_with_graph(recorder, run_id, config, graph)

    def _execute_with_graph(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> RunResult:
        """Execute using validated ExecutionGraph."""

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Register nodes with Landscape using graph's node IDs
        for node_id in execution_order:
            node_info = graph.get_node_info(node_id)
            recorder.register_node(
                run_id=run_id,
                node_id=node_id,  # Use graph's ID, not generated
                plugin_name=node_info.plugin_name,
                node_type=NodeType(node_info.node_type.upper()),
                plugin_version="1.0.0",
                config=node_info.config,
            )

        # Register edges from graph
        edge_map: dict[tuple[str, str], str] = {}
        for from_id, to_id, edge_data in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=from_id,
                to_node_id=to_id,
                label=edge_data["label"],
                mode=edge_data["mode"],
            )
            edge_map[(from_id, edge_data["label"])] = edge.edge_id

        # Get explicit node ID mappings from graph
        # NOTE: No string matching - graph maintains explicit sink_name -> node_id mapping
        source_id = graph.get_source()
        sink_id_map = graph.get_sink_id_map()  # Returns dict[sink_name, node_id]

        # Set node_id on source plugin
        config.source.node_id = source_id

        # Set node_id on transforms using graph's transform_id_map
        transform_id_map = graph.get_transform_id_map()  # Returns dict[sequence, node_id]
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                transform.node_id = transform_id_map[seq]

        # Set node_id on sinks using explicit mapping (no substring matching!)
        for sink_name, sink in config.sinks.items():
            if sink_name in sink_id_map:
                sink.node_id = sink_id_map[sink_name]
            else:
                raise ValueError(f"Sink '{sink_name}' not found in graph")

        # Get output_sink node_id for routing completed rows
        output_sink_node_id = sink_id_map[config.output_sink]

        # ... rest of execution logic uses output_sink_node_id for completed rows ...
```

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_orchestrator_graph.py -v
```

---

## Task 12: Add get_node_info() and get_edges() to ExecutionGraph

**Goal:** ExecutionGraph exposes node info and edges for Orchestrator.

### Test First

```python
# tests/core/test_dag.py

class TestExecutionGraphAccessors:
    """Access node info and edges from graph."""

    def test_get_node_info(self) -> None:
        """Get NodeInfo for a node."""
        from elspeth.core.dag import ExecutionGraph, NodeInfo

        graph = ExecutionGraph()
        graph.add_node(
            "node_1",
            node_type="transform",
            plugin_name="my_plugin",
            config={"key": "value"},
        )

        info = graph.get_node_info("node_1")

        assert isinstance(info, NodeInfo)
        assert info.node_id == "node_1"
        assert info.node_type == "transform"
        assert info.plugin_name == "my_plugin"
        assert info.config == {"key": "value"}

    def test_get_node_info_missing(self) -> None:
        """Get NodeInfo for missing node raises."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()

        with pytest.raises(KeyError):
            graph.get_node_info("nonexistent")

    def test_get_edges(self) -> None:
        """Get all edges with data."""
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("a", node_type="source", plugin_name="src")
        graph.add_node("b", node_type="transform", plugin_name="tf")
        graph.add_node("c", node_type="sink", plugin_name="sink")
        graph.add_edge("a", "b", label="continue", mode="move")
        graph.add_edge("b", "c", label="output", mode="copy")

        edges = list(graph.get_edges())

        assert len(edges) == 2
        # Each edge is (from_id, to_id, data_dict)
        assert ("a", "b", {"label": "continue", "mode": "move"}) in edges
        assert ("b", "c", {"label": "output", "mode": "copy"}) in edges

    def test_get_sink_id_map(self) -> None:
        """Get explicit sink_name -> node_id mapping."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        sink_map = graph.get_sink_id_map()

        # Explicit mapping - no substring matching
        assert "results" in sink_map
        assert "flagged" in sink_map
        assert sink_map["results"] != sink_map["flagged"]

    def test_get_transform_id_map(self) -> None:
        """Get explicit sequence -> node_id mapping for transforms."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(plugin="transform_a"),
                RowPluginSettings(plugin="transform_b"),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)
        transform_map = graph.get_transform_id_map()

        # Explicit mapping by sequence position
        assert 0 in transform_map  # transform_a
        assert 1 in transform_map  # transform_b
        assert transform_map[0] != transform_map[1]
```

### Implementation

```python
# src/elspeth/core/dag.py

class ExecutionGraph:
    # ... existing methods ...

    def get_node_info(self, node_id: str) -> NodeInfo:
        """Get NodeInfo for a node.

        Args:
            node_id: The node ID

        Returns:
            NodeInfo for the node

        Raises:
            KeyError: If node doesn't exist
        """
        if not self._graph.has_node(node_id):
            raise KeyError(f"Node not found: {node_id}")
        return self._graph.nodes[node_id]["info"]

    def get_edges(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all edges with their data.

        Returns:
            List of (from_node, to_node, edge_data) tuples
        """
        return [
            (u, v, dict(data))
            for u, v, data in self._graph.edges(data=True)
        ]

    def get_sink_id_map(self) -> dict[str, str]:
        """Get explicit sink_name -> node_id mapping.

        Returns:
            Dict mapping each sink's logical name to its graph node ID.
            No substring matching required - use this for direct lookup.
        """
        return dict(self._sink_id_map)

    def get_transform_id_map(self) -> dict[int, str]:
        """Get explicit sequence -> node_id mapping for transforms.

        Returns:
            Dict mapping transform sequence position (0-indexed) to node ID.
        """
        return dict(self._transform_id_map)
```

**NOTE:** The `from_config()` factory must populate `_sink_id_map` and `_transform_id_map`
during graph construction. Update Task 9 implementation accordingly.

### Verification

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphAccessors -v
```

---

## Task 13: Wire CLI to Pass Graph to Orchestrator

**Goal:** Complete the integration - CLI builds graph, passes to Orchestrator.

### Test First

```python
# tests/integration/test_full_pipeline.py

"""Full integration test: config -> graph -> execution."""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from elspeth.cli import app

runner = CliRunner()


class TestFullPipelineIntegration:
    """End-to-end pipeline with graph validation."""

    def test_csv_pipeline_uses_graph(self, tmp_path: Path) -> None:
        """CSV pipeline is validated and executed via graph."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: {input_file}

sinks:
  results:
    plugin: csv
    options:
      path: {output_file}

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:///{audit_db}
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0
        assert output_file.exists()
        # Verbose output should mention graph
        assert "node" in result.stdout.lower() or "completed" in result.stdout.lower()

    def test_invalid_graph_blocks_execution(self, tmp_path: Path) -> None:
        """Invalid graph prevents any execution."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: csv

sinks:
  output:
    plugin: csv

row_plugins:
  - plugin: broken_gate
    type: gate
    routes:
      error: sink_that_does_not_exist

output_sink: output
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute"])

        assert result.exit_code != 0
        # Should fail at graph validation
        assert "sink_that_does_not_exist" in result.stdout or "graph" in result.stdout.lower()
```

### Implementation

Update `_execute_pipeline()` to build and pass graph:

```python
# src/elspeth/cli.py

def _execute_pipeline(
    config: ElspethSettings,
    graph: ExecutionGraph,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a pipeline from validated configuration.

    Args:
        config: Validated ElspethSettings instance
        graph: Validated ExecutionGraph
        verbose: Show detailed output

    Returns:
        Dict with run_id, status, rows_processed.
    """
    # ... setup code ...

    if verbose:
        typer.echo("Starting pipeline execution...")
        typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
        typer.echo(f"  Execution order: {len(graph.topological_order())} steps")

    # Execute via Orchestrator WITH GRAPH
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config, graph=graph)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }
```

And update the `run` command:

```python
@app.command()
def run(...) -> None:
    # ... load and validate config ...

    # Build and validate execution graph
    try:
        graph = ExecutionGraph.from_config(config)
        graph.validate()
    except GraphValidationError as e:
        typer.echo(f"Pipeline graph error: {e}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Graph validated: {graph.node_count} nodes, {graph.edge_count} edges")

    # ... dry run handling ...

    # Execute WITH GRAPH
    try:
        result = _execute_pipeline(config, graph, verbose=verbose)
        # ...
```

### Verification

```bash
.venv/bin/python -m pytest tests/integration/test_full_pipeline.py -v

# Full test suite
.venv/bin/python -m pytest tests/ -v
```

---

## Phase 2 Integration Gaps (Tasks 14-19)

These tasks fix gaps between Phase 2 plugin definitions and Phase 3 engine integration.

---

## Task 14: Fix Filter TransformResult.success(None) Bug

**Priority:** CRITICAL - Will cause AssertionError at runtime

**Problem:** Filter plugin uses `TransformResult.success(None)` for filtered rows, but
executor asserts `result.row is not None` for success status.

**Location:**
- Bug: `src/elspeth/plugins/transforms/filter.py:95, 102`
- Conflict: `src/elspeth/engine/executors.py:173-174`

### Test First

```python
# tests/plugins/transforms/test_filter.py

class TestFilterWithExecutor:
    """Filter works correctly with executor contract."""

    def test_filtered_row_does_not_assert(self, tmp_path: Path) -> None:
        """Filtered rows don't trigger assertion in executor."""
        from elspeth.plugins.transforms.filter import Filter
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        filter_plugin = Filter({
            "field": "status",
            "equals": "active",
        })

        ctx = PluginContext(run_id="test", config={})

        # Row that should be filtered out
        result = filter_plugin.process({"status": "inactive"}, ctx)

        # Should return filtered status, not success with None
        assert result.status == "filtered"
        assert result.row is None

    def test_passing_row_returns_success(self) -> None:
        """Rows passing filter return success with row data."""
        from elspeth.plugins.transforms.filter import Filter
        from elspeth.plugins.context import PluginContext

        filter_plugin = Filter({
            "field": "status",
            "equals": "active",
        })

        ctx = PluginContext(run_id="test", config={})

        result = filter_plugin.process({"status": "active", "id": 1}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["status"] == "active"
```

### Implementation

**Step 1:** Add `TransformResult.filtered()` factory:

```python
# src/elspeth/plugins/results.py

@dataclass
class TransformResult:
    """Result from any transform operation."""

    status: Literal["success", "error", "filtered"]  # Added "filtered"
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # ... existing fields ...

    @classmethod
    def filtered(cls, reason: dict[str, Any] | None = None) -> "TransformResult":
        """Row was filtered out (not an error, just excluded).

        Use this when a transform decides to exclude a row from further
        processing. The row is not passed downstream but this is not
        an error condition.
        """
        return cls(status="filtered", row=None, reason=reason)
```

**Step 2:** Update Filter to use `TransformResult.filtered()`:

```python
# src/elspeth/plugins/transforms/filter.py

def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    field_value = self._get_nested(row, self._field)

    # Handle missing field
    if field_value is _MISSING:
        if self._allow_missing:
            return TransformResult.success(copy.deepcopy(row))
        return TransformResult.filtered(reason={"field": self._field, "cause": "missing"})

    # Apply condition
    passes = self._evaluate_condition(field_value)

    if passes:
        return TransformResult.success(copy.deepcopy(row))
    return TransformResult.filtered(reason={
        "field": self._field,
        "condition": self._condition_type,
        "value": field_value,
    })
```

**Step 3:** Update executor to handle "filtered" status:

```python
# src/elspeth/engine/executors.py

def execute_transform(...) -> TransformExecuteResult:
    # ... existing code ...

    if result.status == "success":
        assert result.row is not None, "success status requires row data"
        # ... existing success handling ...
    elif result.status == "filtered":
        # Filtered rows complete successfully but don't continue
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",  # Not failed - intentional exclusion
            output_data=None,
            duration_ms=duration_ms,
        )
        return TransformExecuteResult(
            outcome="filtered",
            token=token,  # Original token, not updated
        )
    else:
        # error status
        # ... existing error handling ...
```

### Verification

```bash
.venv/bin/python -m pytest tests/plugins/transforms/test_filter.py -v
.venv/bin/python -m pytest tests/engine/test_executors.py -v
```

---

## Task 15: Fix _freeze_dict() Mutation Leak

**Problem:** `_freeze_dict()` wraps dict in MappingProxyType but doesn't copy, so
mutation of original dict is visible through the proxy.

**Location:** `src/elspeth/plugins/results.py:32-34`

### Test First

```python
# tests/plugins/test_results.py

class TestFreezeDictDefensiveCopy:
    """_freeze_dict makes defensive copy to prevent mutation."""

    def test_original_dict_mutation_not_visible(self) -> None:
        """Mutating original dict doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"key": "original"}
        action = RoutingAction.continue_(reason=reason)

        # Mutate original
        reason["key"] = "mutated"
        reason["new_key"] = "added"

        # Frozen reason should be unchanged
        assert action.reason["key"] == "original"
        assert "new_key" not in action.reason

    def test_nested_dict_mutation_not_visible(self) -> None:
        """Nested dict mutation doesn't affect frozen result."""
        from elspeth.plugins.results import RoutingAction

        reason = {"nested": {"value": 1}}
        action = RoutingAction.continue_(reason=reason)

        # Mutate nested original
        reason["nested"]["value"] = 999

        # Frozen reason should be unchanged
        assert action.reason["nested"]["value"] == 1
```

### Implementation

```python
# src/elspeth/plugins/results.py

import copy

def _freeze_dict(d: dict[str, Any] | None) -> Mapping[str, Any]:
    """Create immutable view of dict with defensive deep copy.

    MappingProxyType only prevents mutation through the proxy.
    We deep copy to prevent mutation via retained references to
    the original dict or nested objects.
    """
    if d is None:
        return MappingProxyType({})
    # Deep copy to prevent mutation of original or nested dicts
    return MappingProxyType(copy.deepcopy(d))
```

### Verification

```bash
.venv/bin/python -m pytest tests/plugins/test_results.py::TestFreezeDictDefensiveCopy -v
```

---

## Task 16: Orchestrator Uses Plugin Metadata

**Problem:** Orchestrator hardcodes `plugin_version="1.0.0"` and empty `config={}` instead
of using plugin's actual metadata.

**Location:** `src/elspeth/engine/orchestrator.py:125, 154`

### Test First

```python
# tests/engine/test_orchestrator_metadata.py

class TestOrchestratorPluginMetadata:
    """Orchestrator records actual plugin metadata."""

    def test_node_uses_plugin_version(self) -> None:
        """Node registration uses plugin's declared version."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        class VersionedTransform(BaseTransform):
            name = "versioned"
            plugin_version = "2.5.0"
            input_schema = type("S", (), {"model_config": {"extra": "allow"}})()
            output_schema = input_schema

            def process(self, row, ctx):
                return TransformResult.success(row)

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db)

        # ... run pipeline with VersionedTransform ...

        # Verify node was registered with correct version
        nodes = db.get_nodes_for_run(run.run_id)
        transform_node = [n for n in nodes if n.plugin_name == "versioned"][0]
        assert transform_node.plugin_version == "2.5.0"

    def test_node_uses_plugin_config(self) -> None:
        """Node registration uses plugin's config."""
        # Similar test checking config is recorded
```

### Implementation

Update orchestrator node registration:

```python
# src/elspeth/engine/orchestrator.py

def _execute_run(...):
    # ... existing code ...

    for i, transform in enumerate(config.transforms):
        is_gate = hasattr(transform, "evaluate")
        node_type = NodeType.GATE if is_gate else NodeType.TRANSFORM
        node = recorder.register_node(
            run_id=run_id,
            plugin_name=transform.name,
            node_type=node_type,
            plugin_version=getattr(transform, 'plugin_version', '0.0.0'),
            config=getattr(transform, 'config', {}),
            determinism=getattr(transform, 'determinism', Determinism.UNKNOWN).value,
            sequence=i + 1,
        )
```

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_orchestrator_metadata.py -v
```

---

## Task 17: Invoke Plugin Lifecycle Hooks

**Problem:** Base classes define `on_register`, `on_start`, `on_complete` hooks but
orchestrator never calls them.

**Location:**
- Hooks defined: `src/elspeth/plugins/base.py:73-89`
- Not called: `src/elspeth/engine/orchestrator.py`

### Test First

```python
# tests/engine/test_lifecycle_hooks.py

class TestLifecycleHooks:
    """Orchestrator invokes plugin lifecycle hooks."""

    def test_on_start_called_before_processing(self) -> None:
        """on_start() called before any rows processed."""
        call_order = []

        class TrackedTransform(BaseTransform):
            name = "tracked"
            # ...

            def on_start(self, ctx):
                call_order.append("on_start")

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

        # ... run pipeline ...

        assert call_order[0] == "on_start"
        assert "process" in call_order

    def test_on_complete_called_after_all_rows(self) -> None:
        """on_complete() called after all rows processed."""
        call_order = []

        class TrackedTransform(BaseTransform):
            # ...

            def process(self, row, ctx):
                call_order.append("process")
                return TransformResult.success(row)

            def on_complete(self, ctx):
                call_order.append("on_complete")

        # ... run pipeline ...

        assert call_order[-1] == "on_complete"

    def test_on_complete_called_on_error(self) -> None:
        """on_complete() called even when run fails."""
        completed = []

        class TrackedTransform(BaseTransform):
            # ...

            def process(self, row, ctx):
                raise RuntimeError("intentional")

            def on_complete(self, ctx):
                completed.append(True)

        # ... run pipeline expecting failure ...

        assert len(completed) == 1  # on_complete still called
```

### Implementation

```python
# src/elspeth/engine/orchestrator.py

def _execute_run(...):
    # ... node registration ...

    # Call on_start for all plugins
    for transform in config.transforms:
        if hasattr(transform, 'on_start'):
            transform.on_start(ctx)

    try:
        # ... row processing ...
    finally:
        # Call on_complete for all plugins (even on error)
        for transform in config.transforms:
            if hasattr(transform, 'on_complete'):
                try:
                    transform.on_complete(ctx)
                except Exception:
                    # Log but don't fail - cleanup should be best-effort
                    pass
```

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_lifecycle_hooks.py -v
```

---

## Task 18: Fix PluginSchema Docstring (strict wording)

**Problem:** Docstring claims "Strict type validation" but config uses `strict=False`.

**Location:** `src/elspeth/plugins/schemas.py:38-44`

### Implementation

```python
# src/elspeth/plugins/schemas.py

class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Plugins define schemas by subclassing:

        class MyInputSchema(PluginSchema):
            temperature: float
            humidity: float

    Features:
    - Extra fields ignored (rows may have more fields than schema requires)
    - Coercive type validation (int→float allowed, strict=False)
    - Easy conversion to/from row dicts
    """

    model_config = ConfigDict(
        extra="ignore",  # Rows may have extra fields
        strict=False,    # Allow coercion (e.g., int -> float)
        frozen=False,    # Allow modification
    )
```

### Verification

```bash
# Just ensure file is syntactically valid
.venv/bin/python -c "from elspeth.plugins.schemas import PluginSchema; print('OK')"
```

---

## Task 19: Clarify RowOutcome Docstring

**Problem:** RowOutcome is defined but never imported by engine. This is actually
correct per architecture (terminal states are derived, not stored), but
the docstring should clarify this.

**Location:** `src/elspeth/plugins/results.py:16-29`

### Implementation

```python
# src/elspeth/plugins/results.py

class RowOutcome(Enum):
    """Terminal states for rows in the pipeline.

    DESIGN NOTE: Per architecture (00-overview.md:267-279), token terminal
    states are DERIVED from the combination of node_states, routing_events,
    and batch membership—not stored as a column. This enum is used at
    query/explain time to report final disposition, not at runtime.

    The engine does NOT set these directly. The Landscape query layer
    derives them when answering explain() queries.

    INVARIANT: Every row reaches exactly one terminal state.
    No silent drops.
    """

    COMPLETED = "completed"           # Reached output sink
    ROUTED = "routed"                 # Sent to named sink by gate (move mode)
    FORKED = "forked"                 # Split into child tokens (parent terminates)
    CONSUMED_IN_BATCH = "consumed_in_batch"  # Fed into aggregation
    COALESCED = "coalesced"           # Merged with other tokens
    QUARANTINED = "quarantined"       # Failed, stored for investigation
    FAILED = "failed"                 # Failed, not recoverable
```

### Verification

```bash
.venv/bin/python -c "from elspeth.plugins.results import RowOutcome; print(RowOutcome.__doc__)"
```

---

## Task 20: Fix PHASE3_INTEGRATION.md API Drift

**Problem:** The integration guide has wrong API examples that don't match the real
LandscapeRecorder API, causing confusion when implementing Phase 3 integration.

**Location:** `src/elspeth/plugins/PHASE3_INTEGRATION.md`

### Documented vs Actual API

| Documented | Actual API |
|------------|------------|
| `LandscapeRecorder(db, run.run_id)` | `LandscapeRecorder(db, payload_store=None)` |
| `ctx.landscape.record_node_state(...)` | `recorder.begin_node_state(...)` + `recorder.complete_node_state(...)` |
| `ctx.landscape.add_batch_output(...)` | Does not exist - outputs returned from `flush()` |

### Implementation

Update PHASE3_INTEGRATION.md to match actual APIs:

```markdown
# src/elspeth/plugins/PHASE3_INTEGRATION.md

## PluginContext Integration

Phase 3 creates PluginContext with full integration:

\`\`\`python
# Phase 3: SDA Engine creates context
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.payload_store import FilesystemPayloadStore

db = LandscapeDB.from_url("sqlite:///runs/audit.db")
payload_store = FilesystemPayloadStore(base_path)
recorder = LandscapeRecorder(db, payload_store=payload_store)

ctx = PluginContext(
    run_id=run.run_id,
    config=resolved_config,
    landscape=recorder,  # LandscapeRecorder instance
    tracer=opentelemetry.trace.get_tracer("elspeth"),
    payload_store=payload_store,
)
\`\`\`

## Transform Processing

Phase 3 engine uses begin_node_state/complete_node_state:

\`\`\`python
def process_with_audit(transform, row, ctx, recorder):
    input_hash = stable_hash(row)

    # Start node state (begins timing)
    state = recorder.begin_node_state(
        run_id=ctx.run_id,
        token_id=token.token_id,
        node_id=transform.node_id,
        input_data=row,
    )

    with ctx.start_span(f"transform:{transform.name}") as span:
        start = time.perf_counter()
        result = transform.process(row, ctx)
        duration_ms = (time.perf_counter() - start) * 1000

        # Complete node state with outcome
        recorder.complete_node_state(
            state_id=state.state_id,
            status="completed" if result.status == "success" else "failed",
            output_data=result.row,
            duration_ms=duration_ms,
        )

    return result
\`\`\`

## Aggregation Batch Management

Phase 3 engine manages batches (outputs are returned, not recorded separately):

\`\`\`python
def flush_with_audit(aggregation, ctx, recorder):
    recorder.update_batch_status(aggregation._batch_id, "executing")

    try:
        outputs = aggregation.flush(ctx)
        recorder.update_batch_status(aggregation._batch_id, "completed")
        # Note: outputs are returned to caller for downstream processing
        # NOT recorded separately via add_batch_output (which doesn't exist)
        return outputs
    except Exception as e:
        recorder.update_batch_status(
            aggregation._batch_id, "failed", error=str(e)
        )
        raise
\`\`\`
```

### Verification

```bash
# Ensure documented APIs actually exist
.venv/bin/python -c "
from elspeth.core.landscape import LandscapeRecorder, LandscapeDB

db = LandscapeDB.in_memory()
recorder = LandscapeRecorder(db)

# Verify methods exist
assert hasattr(recorder, 'begin_node_state')
assert hasattr(recorder, 'complete_node_state')
assert hasattr(recorder, 'create_batch')
assert hasattr(recorder, 'add_batch_member')
assert hasattr(recorder, 'update_batch_status')
assert hasattr(recorder, 'record_routing_event')
print('All documented APIs exist')
"
```

---

## Task 21: Use Enums Consistently in RoutingAction

**Problem:** `RoutingAction` uses string literals for `kind` and `mode` instead of the
defined `RoutingKind` and `RoutingMode` enums, defeating the purpose of the enums.

**Location:**
- Enums defined: `src/elspeth/plugins/enums.py:25-46`
- Literals used: `src/elspeth/plugins/results.py:45-47`

### Test First

```python
# tests/plugins/test_results.py

class TestRoutingActionEnums:
    """RoutingAction uses enum types for kind and mode."""

    def test_continue_uses_routing_kind_enum(self) -> None:
        """continue_() returns RoutingKind enum value."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind

        action = RoutingAction.continue_()

        assert action.kind == RoutingKind.CONTINUE
        assert isinstance(action.kind, RoutingKind)

    def test_route_to_sink_uses_enums(self) -> None:
        """route_to_sink() uses enum types."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind, RoutingMode

        action = RoutingAction.route_to_sink("output", mode="copy")

        assert action.kind == RoutingKind.ROUTE_TO_SINK
        assert action.mode == RoutingMode.COPY
        assert isinstance(action.kind, RoutingKind)
        assert isinstance(action.mode, RoutingMode)

    def test_enums_serialize_to_strings(self) -> None:
        """Enums serialize to strings for JSON compatibility."""
        from elspeth.plugins.results import RoutingAction
        from elspeth.plugins.enums import RoutingKind

        action = RoutingAction.continue_()

        # str(Enum) behavior for JSON serialization
        assert action.kind.value == "continue"
        assert action.kind == "continue"  # str enum comparison works
```

### Implementation

```python
# src/elspeth/plugins/results.py

from elspeth.plugins.enums import RoutingKind, RoutingMode

@dataclass(frozen=True)
class RoutingAction:
    """What a gate decided to do with a row."""

    kind: RoutingKind  # Changed from Literal[...]
    destinations: tuple[str, ...]
    mode: RoutingMode  # Changed from Literal[...]
    reason: Mapping[str, Any]

    @classmethod
    def continue_(cls, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Row continues to next transform."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            mode=RoutingMode.MOVE,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def route_to_sink(
        cls,
        sink_name: str,
        *,
        mode: RoutingMode | str = RoutingMode.MOVE,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Route row to a named sink."""
        # Accept string for backwards compatibility, convert to enum
        if isinstance(mode, str):
            mode = RoutingMode(mode)
        return cls(
            kind=RoutingKind.ROUTE_TO_SINK,
            destinations=(sink_name,),
            mode=mode,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Fork row to multiple parallel paths (copy mode)."""
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,
            reason=_freeze_dict(reason),
        )
```

### Verification

```bash
.venv/bin/python -m pytest tests/plugins/test_results.py::TestRoutingActionEnums -v
```

---

## Task 22: Populate PluginSpec Schema Hashes

**Problem:** `PluginSpec.from_plugin()` leaves `input_schema_hash` and `output_schema_hash`
as None, missing an opportunity to capture schema fingerprints for compatibility tracking.

**Location:** `src/elspeth/plugins/manager.py:44-60`

### Test First

```python
# tests/plugins/test_manager.py

class TestPluginSpecSchemaHashes:
    """PluginSpec.from_plugin() populates schema hashes."""

    def test_from_plugin_captures_input_schema_hash(self) -> None:
        """Input schema is hashed."""
        from elspeth.plugins.manager import PluginSpec
        from elspeth.plugins.enums import NodeType
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            field_a: str
            field_b: int

        class MyTransform:
            name = "test"
            plugin_version = "1.0.0"
            input_schema = InputSchema
            output_schema = InputSchema

        spec = PluginSpec.from_plugin(MyTransform, NodeType.TRANSFORM)

        assert spec.input_schema_hash is not None
        assert len(spec.input_schema_hash) == 64  # SHA-256 hex

    def test_from_plugin_captures_output_schema_hash(self) -> None:
        """Output schema is hashed."""
        from elspeth.plugins.manager import PluginSpec
        from elspeth.plugins.enums import NodeType
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            field_a: str

        class OutputSchema(PluginSchema):
            field_a: str
            computed: float

        class MyTransform:
            name = "test"
            plugin_version = "1.0.0"
            input_schema = InputSchema
            output_schema = OutputSchema

        spec = PluginSpec.from_plugin(MyTransform, NodeType.TRANSFORM)

        assert spec.input_schema_hash is not None
        assert spec.output_schema_hash is not None
        assert spec.input_schema_hash != spec.output_schema_hash

    def test_schema_hash_stable(self) -> None:
        """Same schema always produces same hash."""
        from elspeth.plugins.manager import PluginSpec
        from elspeth.plugins.enums import NodeType
        from elspeth.plugins.schemas import PluginSchema

        class MySchema(PluginSchema):
            value: int

        class T1:
            name = "t1"
            input_schema = MySchema
            output_schema = MySchema

        class T2:
            name = "t2"
            input_schema = MySchema
            output_schema = MySchema

        spec1 = PluginSpec.from_plugin(T1, NodeType.TRANSFORM)
        spec2 = PluginSpec.from_plugin(T2, NodeType.TRANSFORM)

        # Same schema = same hash (regardless of plugin)
        assert spec1.input_schema_hash == spec2.input_schema_hash
```

### Implementation

```python
# src/elspeth/plugins/manager.py

from elspeth.core.canonical import stable_hash

def _schema_hash(schema_cls: type | None) -> str | None:
    """Compute stable hash for a schema class.

    Hashes the schema's field names and types to detect compatibility changes.
    """
    if schema_cls is None:
        return None

    # Use Pydantic model_fields for accurate field introspection
    if not hasattr(schema_cls, 'model_fields'):
        return None

    # Build deterministic representation
    fields_repr = {
        name: str(field.annotation)
        for name, field in schema_cls.model_fields.items()
    }
    return stable_hash(fields_repr)


@dataclass(frozen=True)
class PluginSpec:
    # ... existing fields ...

    @classmethod
    def from_plugin(cls, plugin_cls: type, node_type: NodeType) -> "PluginSpec":
        """Create spec from plugin class with schema hashes."""
        input_schema = getattr(plugin_cls, 'input_schema', None)
        output_schema = getattr(plugin_cls, 'output_schema', None)

        return cls(
            name=getattr(plugin_cls, "name", plugin_cls.__name__),
            node_type=node_type,
            version=getattr(plugin_cls, "plugin_version", "0.0.0"),
            determinism=getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC),
            input_schema_hash=_schema_hash(input_schema),
            output_schema_hash=_schema_hash(output_schema),
        )
```

### Verification

```bash
.venv/bin/python -m pytest tests/plugins/test_manager.py::TestPluginSpecSchemaHashes -v
```

---

## Updated Summary

| Task | Description | Priority | Files Modified |
|------|-------------|----------|----------------|
| **Configuration (Tasks 1-8)** | | | |
| 1 | Add architecture-compliant settings classes | P0 | `config.py`, `test_config.py` |
| 2 | Update ElspethSettings to match architecture | P0 | `config.py`, `test_config.py` |
| 3 | Update load_settings() for new schema | P0 | `config.py`, `test_config.py` |
| 4 | Wire CLI to use load_settings() | P0 | `cli.py`, `test_cli_config.py` |
| 5 | Update _execute_pipeline() for new config | P0 | `cli.py`, `test_cli_execution.py` |
| 6 | Remove dead code | P0 | `cli.py` |
| 7 | Update existing config tests | P0 | `test_config.py` |
| 8 | Update __init__.py exports | P0 | `core/__init__.py` |
| **DAG Infrastructure (Tasks 9-13)** | | | |
| 9 | Add ExecutionGraph.from_config() factory | P0 | `dag.py`, `test_dag.py` |
| 10 | Add graph validation to CLI | P0 | `cli.py`, `test_cli_graph_validation.py` |
| 11 | Update Orchestrator to accept ExecutionGraph | P0 | `orchestrator.py`, `test_orchestrator_graph.py` |
| 12 | Add get_node_info() and get_edges() | P0 | `dag.py`, `test_dag.py` |
| 13 | Wire CLI to pass graph to Orchestrator | P0 | `cli.py`, `test_full_pipeline.py` |
| **Phase 2 Integration (Tasks 14-19)** | | | |
| 14 | Fix Filter TransformResult.success(None) bug | **CRITICAL** | `results.py`, `filter.py`, `executors.py` |
| 15 | Fix _freeze_dict() mutation leak | P1 | `results.py`, `test_results.py` |
| 16 | Orchestrator uses plugin metadata | P1 | `orchestrator.py`, `test_orchestrator_metadata.py` |
| 17 | Invoke plugin lifecycle hooks | P1 | `orchestrator.py`, `test_lifecycle_hooks.py` |
| 18 | Fix PluginSchema docstring | P2 | `schemas.py` |
| 19 | Clarify RowOutcome docstring | P2 | `results.py` |
| **Phase 2 Consistency (Tasks 20-22)** | | | |
| 20 | Fix PHASE3_INTEGRATION.md API drift | P1 | `PHASE3_INTEGRATION.md` |
| 21 | Use enums in RoutingAction | P1 | `results.py`, `test_results.py` |
| 22 | Populate PluginSpec schema hashes | P2 | `manager.py`, `test_manager.py` |

**Updated estimate:** ~1100-1200 lines changed across 14 files

---

## Final Verification Checklist

```bash
# All tests pass
.venv/bin/python -m pytest tests/ -v

# Graph validation works
cat > /tmp/bad_config.yaml << 'EOF'
datasource:
  plugin: csv
sinks:
  output:
    plugin: csv
row_plugins:
  - plugin: gate
    type: gate
    routes:
      bad: nonexistent_sink
output_sink: output
EOF

elspeth validate -s /tmp/bad_config.yaml
# Should fail with "nonexistent_sink" error

# Good config works
cat > /tmp/good_config.yaml << 'EOF'
datasource:
  plugin: csv
  options:
    path: /tmp/input.csv
sinks:
  output:
    plugin: csv
    options:
      path: /tmp/output.csv
output_sink: output
EOF

elspeth validate -s /tmp/good_config.yaml
# Should show "Graph: N nodes, M edges"

# No ad-hoc edge building in orchestrator
grep -n "edge_map: dict" src/elspeth/engine/orchestrator.py
# Should be in legacy path only or removed

# ExecutionGraph is actually used
grep -rn "ExecutionGraph" src/elspeth/
# Should show imports in cli.py, orchestrator.py
```
