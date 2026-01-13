# Configuration System Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the configuration system with the architecture specification so that CLI uses validated Pydantic settings with architecture-compliant field names.

**Architecture:** The current implementation has three incompatible config formats (README, CLI, config.py). This plan consolidates them into a single Pydantic-validated schema using `datasource`, `sinks`, `row_plugins`, `output_sink`, `landscape`, and `concurrency` fields. The CLI will use `load_settings()` exclusively instead of ad-hoc `yaml.safe_load()`.

**Tech Stack:** Pydantic v2, Dynaconf, pytest, typer

---

## Task 1: Add DatasourceSettings Class

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestDatasourceSettings:
    """DatasourceSettings matches architecture specification."""

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

    def test_datasource_settings_frozen(self) -> None:
        """DatasourceSettings is immutable."""
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import DatasourceSettings

        ds = DatasourceSettings(plugin="csv")
        with pytest.raises(ValidationError):
            ds.plugin = "json"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestDatasourceSettings -v`
Expected: FAIL with `cannot import name 'DatasourceSettings'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/config.py` after the imports:

```python
from typing import Any


class DatasourceSettings(BaseModel):
    """Source plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestDatasourceSettings -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): add DatasourceSettings class

Adds architecture-compliant DatasourceSettings with plugin name and
options dict for source configuration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add RowPluginSettings Class

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestRowPluginSettings:
    """RowPluginSettings matches architecture specification."""

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
        assert rp.options == {"field": "confidence", "min": 0.8}
        assert rp.routes == {"pass": "continue", "fail": "quarantine"}

    def test_row_plugin_settings_defaults(self) -> None:
        """RowPluginSettings defaults: type=transform, no routes."""
        from elspeth.core.config import RowPluginSettings

        rp = RowPluginSettings(plugin="field_mapper")
        assert rp.type == "transform"
        assert rp.options == {}
        assert rp.routes is None

    def test_row_plugin_settings_type_validation(self) -> None:
        """Type must be 'transform' or 'gate'."""
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError):
            RowPluginSettings(plugin="test", type="invalid")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestRowPluginSettings -v`
Expected: FAIL with `cannot import name 'RowPluginSettings'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/config.py`:

```python
from typing import Any, Literal


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
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestRowPluginSettings -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): add RowPluginSettings class

Adds architecture-compliant RowPluginSettings for transform/gate
configuration with plugin, type, options, and routes fields.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add SinkSettings Class

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestSinkSettings:
    """SinkSettings matches architecture specification."""

    def test_sink_settings_structure(self) -> None:
        """SinkSettings has plugin and options."""
        from elspeth.core.config import SinkSettings

        sink = SinkSettings(plugin="csv", options={"path": "output/results.csv"})
        assert sink.plugin == "csv"
        assert sink.options == {"path": "output/results.csv"}

    def test_sink_settings_options_default_empty(self) -> None:
        """Options defaults to empty dict."""
        from elspeth.core.config import SinkSettings

        sink = SinkSettings(plugin="database")
        assert sink.options == {}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestSinkSettings -v`
Expected: FAIL with `cannot import name 'SinkSettings'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/config.py`:

```python
class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestSinkSettings -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): add SinkSettings class

Adds architecture-compliant SinkSettings with plugin name and options.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add LandscapeSettings Class

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestLandscapeSettings:
    """LandscapeSettings matches architecture specification."""

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
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import LandscapeSettings

        with pytest.raises(ValidationError):
            LandscapeSettings(backend="mysql")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestLandscapeSettings -v`
Expected: FAIL with `cannot import name 'LandscapeSettings'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/config.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestLandscapeSettings -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): add LandscapeSettings class

Adds architecture-compliant LandscapeSettings with enabled, backend,
and url fields. Uses str for url to preserve PostgreSQL DSNs.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add ConcurrencySettings Class

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestConcurrencySettings:
    """ConcurrencySettings matches architecture specification."""

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
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import ConcurrencySettings

        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=0)
        with pytest.raises(ValidationError):
            ConcurrencySettings(max_workers=-1)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestConcurrencySettings -v`
Expected: FAIL with `cannot import name 'ConcurrencySettings'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/config.py`:

```python
class ConcurrencySettings(BaseModel):
    """Parallel processing configuration per architecture."""

    model_config = {"frozen": True}

    max_workers: int = Field(
        default=4,
        gt=0,
        description="Maximum parallel workers (default 4, production typically 16)",
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestConcurrencySettings -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): add ConcurrencySettings class

Adds architecture-compliant ConcurrencySettings with max_workers field.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Replace ElspethSettings with Architecture-Compliant Schema

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
class TestElspethSettingsArchitecture:
    """Top-level settings matches architecture specification."""

    def test_elspeth_settings_required_fields(self) -> None:
        """ElspethSettings requires datasource, sinks, output_sink."""
        import pytest
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
            DatasourceSettings,
            ElspethSettings,
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

    def test_elspeth_settings_output_sink_must_exist(self) -> None:
        """output_sink must reference a defined sink."""
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
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
        import pytest
        from pydantic import ValidationError
        from elspeth.core.config import DatasourceSettings, ElspethSettings

        with pytest.raises(ValidationError) as exc_info:
            ElspethSettings(
                datasource=DatasourceSettings(plugin="csv"),
                sinks={},  # Empty!
                output_sink="results",
            )

        assert "sink" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestElspethSettingsArchitecture -v`
Expected: FAIL (tests will fail because ElspethSettings has old schema)

**Step 3: Write minimal implementation**

Replace the `ElspethSettings` class in `src/elspeth/core/config.py`:

```python
from pydantic import BaseModel, Field, field_validator, model_validator


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

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestElspethSettingsArchitecture -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py src/elspeth/core/config.py
git commit -m "$(cat <<'EOF'
feat(config): replace ElspethSettings with architecture-compliant schema

BREAKING: ElspethSettings now requires datasource, sinks, output_sink
instead of the old database-only schema. This aligns with the
architecture specification.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update load_settings() and Add YAML Loading Tests

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

Add to `tests/core/test_config.py`:

```python
from pathlib import Path


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

    def test_load_invalid_output_sink(self, tmp_path: Path) -> None:
        """Error when output_sink doesn't exist."""
        import pytest
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
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestLoadSettingsArchitecture -v`
Expected: FAIL (load_settings uses old schema)

**Step 3: The implementation is already correct**

The `load_settings()` function should already work with the new schema since it converts dynaconf dict to kwargs and passes to `ElspethSettings(**raw_config)`. No changes needed to the function itself.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py::TestLoadSettingsArchitecture -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tests/core/test_config.py
git commit -m "$(cat <<'EOF'
test(config): add load_settings tests for architecture-compliant YAML

Tests cover README example, minimal config, and validation errors.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire CLI run Command to Use load_settings()

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_run_command.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_run_command.py`:

```python
class TestRunCommandWithNewConfig:
    """Run command uses load_settings() for config."""

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
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        # Should fail - 'source' is not valid, must be 'datasource'
        assert result.exit_code != 0

    def test_run_shows_pydantic_errors(self, tmp_path: Path) -> None:
        """Run shows Pydantic validation errors clearly."""
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

        result = runner.invoke(app, ["run", "-s", str(config_file), "--dry-run"])

        assert result.exit_code != 0
        # Should show helpful error messages
        output = result.stdout + (result.stderr or "")
        assert "output_sink" in output.lower() or "nonexistent" in output.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_run_command.py::TestRunCommandWithNewConfig -v`
Expected: FAIL (CLI uses yaml.safe_load, not load_settings)

**Step 3: Write minimal implementation**

Update `src/elspeth/cli.py` - replace the `run` command:

```python
from pydantic import ValidationError

from elspeth.core.config import ElspethSettings, load_settings


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
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_run_command.py::TestRunCommandWithNewConfig -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_run_command.py
git commit -m "$(cat <<'EOF'
feat(cli): wire run command to use load_settings()

Replaces ad-hoc yaml.safe_load with Pydantic-validated load_settings().
Shows clear validation errors on config problems.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wire CLI validate Command to Use load_settings()

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_validate_command.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_validate_command.py`:

```python
class TestValidateCommandWithNewConfig:
    """Validate command uses load_settings() for config."""

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
  url: sqlite:///./audit.db
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
""")

        result = runner.invoke(app, ["validate", "-s", str(config_file)])

        assert result.exit_code != 0
        output = result.stdout + (result.stderr or "")
        assert "output_sink" in output.lower() or "nonexistent" in output.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_validate_command.py::TestValidateCommandWithNewConfig -v`
Expected: FAIL (validate command uses old validation)

**Step 3: Write minimal implementation**

Update `src/elspeth/cli.py` - replace the `validate` command:

```python
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

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_validate_command.py::TestValidateCommandWithNewConfig -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_validate_command.py
git commit -m "$(cat <<'EOF'
feat(cli): wire validate command to use load_settings()

Replaces ad-hoc validation with Pydantic-validated load_settings().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Remove Dead Code from CLI

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Identify dead code**

Search for old validation code that's no longer needed:
- `_validate_config()` function
- `KNOWN_SOURCES` constant
- `KNOWN_SINKS` constant
- `import yaml` (if no longer used elsewhere)

**Step 2: Run grep to confirm these exist**

Run: `grep -n "_validate_config\|KNOWN_SOURCES\|KNOWN_SINKS\|yaml.safe_load" src/elspeth/cli.py`

**Step 3: Delete the dead code**

Remove from `src/elspeth/cli.py`:
- The `_validate_config()` function
- The `KNOWN_SOURCES` and `KNOWN_SINKS` constants
- The `import yaml` line (if unused)

**Step 4: Verify no references remain**

Run: `grep -r "_validate_config\|KNOWN_SOURCES\|KNOWN_SINKS" src/elspeth/`
Expected: No output (nothing found)

**Step 5: Run all CLI tests to ensure nothing broke**

Run: `.venv/bin/python -m pytest tests/cli/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): remove dead config validation code

Removes _validate_config(), KNOWN_SOURCES, KNOWN_SINKS - replaced by
Pydantic validation in load_settings().

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update Existing Config Tests for New Schema

**Files:**
- Modify: `tests/core/test_config.py`

**Step 1: Run existing config tests to find failures**

Run: `.venv/bin/python -m pytest tests/core/test_config.py -v`
Expected: Some tests fail due to old schema expectations

**Step 2: Update tests that use old schema**

Find tests that create `ElspethSettings` with `database=` and update them to use the new schema:

```python
# Before (old schema):
settings = ElspethSettings(
    database=DatabaseSettings(url="sqlite:///test.db")
)

# After (new schema):
settings = ElspethSettings(
    datasource=DatasourceSettings(plugin="csv"),
    sinks={"output": SinkSettings(plugin="csv")},
    output_sink="output",
)
```

**Step 3: Run tests to verify fixes**

Run: `.venv/bin/python -m pytest tests/core/test_config.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/core/test_config.py
git commit -m "$(cat <<'EOF'
test(config): update existing tests for new schema

Migrates tests from old database-only schema to architecture-compliant
datasource/sinks/output_sink schema.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update __init__.py Exports

**Files:**
- Modify: `src/elspeth/core/__init__.py`

**Step 1: Write the failing test**

```python
def test_core_exports_new_settings() -> None:
    """Core module exports new settings classes."""
    from elspeth.core import (
        ConcurrencySettings,
        DatasourceSettings,
        ElspethSettings,
        LandscapeSettings,
        RowPluginSettings,
        SinkSettings,
        load_settings,
    )

    # Just verify they're importable
    assert DatasourceSettings is not None
    assert RowPluginSettings is not None
    assert SinkSettings is not None
    assert LandscapeSettings is not None
    assert ConcurrencySettings is not None
    assert ElspethSettings is not None
    assert load_settings is not None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -c "from elspeth.core import DatasourceSettings"`
Expected: ImportError

**Step 3: Update exports**

Update `src/elspeth/core/__init__.py`:

```python
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

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -c "from elspeth.core import DatasourceSettings, ElspethSettings, load_settings; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add src/elspeth/core/__init__.py
git commit -m "$(cat <<'EOF'
feat(core): export new settings classes from core module

Exports DatasourceSettings, RowPluginSettings, SinkSettings,
LandscapeSettings, ConcurrencySettings for public API.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final Verification

**Step 1: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

**Step 2: Verify README example parses correctly**

```bash
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

.venv/bin/python -m elspeth validate -s /tmp/test.yaml
```

Expected: `Configuration valid: test.yaml`

**Step 3: Verify no dead code remains**

Run: `grep -r "yaml.safe_load\|_validate_config\|KNOWN_SOURCES" src/elspeth/cli.py`
Expected: No output

**Step 4: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/core/config.py src/elspeth/cli.py`
Expected: Success

**Step 5: Commit verification results (optional)**

No commit needed - this is verification only.

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Add DatasourceSettings class | `config.py`, `test_config.py` |
| 2 | Add RowPluginSettings class | `config.py`, `test_config.py` |
| 3 | Add SinkSettings class | `config.py`, `test_config.py` |
| 4 | Add LandscapeSettings class | `config.py`, `test_config.py` |
| 5 | Add ConcurrencySettings class | `config.py`, `test_config.py` |
| 6 | Replace ElspethSettings with new schema | `config.py`, `test_config.py` |
| 7 | Add load_settings YAML tests | `test_config.py` |
| 8 | Wire CLI run command to load_settings | `cli.py`, `test_run_command.py` |
| 9 | Wire CLI validate command to load_settings | `cli.py`, `test_validate_command.py` |
| 10 | Remove dead code from CLI | `cli.py` |
| 11 | Update existing config tests | `test_config.py` |
| 12 | Update __init__.py exports | `core/__init__.py` |
| 13 | Final verification | (verification only) |

**Estimated total:** ~500-600 lines changed across 5 files
