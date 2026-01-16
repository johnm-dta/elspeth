# Code Quality Burn-Down Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Systematically eliminate all swallowed exceptions, defensive coding violations, and integration seam mismatches to achieve zero code quality debt.

**Architecture:** Four-phase approach: (1) Detection infrastructure and baseline, (2) Architectural bug fixes, (3) Systematic pattern remediation with TDD, (4) Verification and CI lock-down.

**Tech Stack:** Python 3.11+, pytest, mypy, ruff, Pydantic, AST-based static analysis

---

## Current State Assessment

**Violations Detected:** 108 total (raw count - see adjustments below)

| Rule | Count | Description |
|------|-------|-------------|
| R1 (dict.get) | 88 | Potential silent defaulting |
| R2 (getattr with default) | 11 | Attribute access with fallback |
| R3 (hasattr) | 7 | Existence checks hiding missing attributes |
| R4 (broad except) | 2 | Exception swallowing |

**Adjusted Violation Count After Review:**

| Category | Raw | Adjustment | Actual | Reason |
|----------|-----|------------|--------|--------|
| TUI trust boundary | 35 | -35 | 0 | Legitimate graceful degradation |
| PluginManager lookups | ~15 | -15 | 0 | Intentional `Optional[T]` API design |
| Orchestrator `suppress()` | 1 | -1 | 0 | Cleanup isolation (whitelist) |
| **Actionable violations** | - | - | **~57** | Requires remediation |

**Open P1 Bugs:** 5

1. Source/Sink lifecycle hooks never called
2. Run configuration not recorded in Landscape
3. Node metadata hardcoded (version/determinism)
4. Defensive whitelist review needed
5. Non-file sink artifact metadata crashes

**Hotspot Files:**

| File | Raw Violations | After Review | Notes |
|------|----------------|--------------|-------|
| tui/widgets/node_detail.py | 19 | 0 | Trust boundary - whitelist |
| plugins/manager.py | 18 | ~3 | Most are Optional API design |
| tui/widgets/lineage_tree.py | 16 | 0 | Trust boundary - whitelist |
| cli.py | 5 | 5 | Config parsing - fix |
| engine/adapters.py | 5 | 5 | Artifact handling - fix |

---

## Review Findings (Code Review + Architecture Alignment)

### Violations Correctly Identified

- **P1 bugs are real integration seam failures** - lifecycle hooks, config recording, metadata
- **Plugin config `dict.get()` patterns** are defensive coding that should use typed Pydantic configs
- **`hasattr` checks in orchestrator** exist because lifecycle was incorrectly optional

### Violations Misclassified (NOT bugs)

1. **PluginManager lookup methods** - `get_source_by_name()` returning `None` is intentional API:
   ```python
   def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
       return self._sources.get(name)  # Correct - Optional return type
   ```

2. **Orchestrator cleanup `suppress(Exception)`** - Intentional isolation:
   ```python
   # One plugin failure shouldn't prevent other plugins from cleanup
   with suppress(Exception):
       transform.on_complete(ctx)
   ```

### Schema Migration Required

The `artifacts` table currently has `path TEXT NOT NULL`. To support database and webhook sinks:
- Rename to `path_or_uri TEXT NOT NULL`
- Accept URI schemes: `file://`, `db://`, `webhook://`

---

## Phase 1: Detection Infrastructure

### Task 1.1: Enable CI Enforcement (Allowlist Mode)

**Files:**
- Create: `config/cicd/no_bug_hiding.yaml`
- Modify: `.github/workflows/ci.yaml` (or equivalent)

**Step 1: Create initial allowlist with ALL current violations**

```yaml
# config/cicd/no_bug_hiding.yaml
version: 1
defaults:
  fail_on_stale: true
  fail_on_expired: true

# WARNING: This is the burn-down list. Every entry must be removed or justified.
# Entries expire 2026-02-15 - must be fixed or explicitly whitelisted by then.
allow_hits: []  # Will be populated by Step 2
```

**Step 2: Generate allowlist entries for all current violations**

Run:
```bash
uv run python scripts/cicd/no_bug_hiding.py check --root src/elspeth --format json \
  | uv run python -c "
import json, sys
from datetime import date, timedelta
d = json.load(sys.stdin)
expires = (date.today() + timedelta(days=30)).isoformat()
print('allow_hits:')
for v in d['violations']:
    print(f'''  - key: \"{v['key']}\"
    owner: \"burn-down\"
    reason: \"Pre-existing - needs review\"
    safety: \"Pending remediation\"
    expires: \"{expires}\"''')
"
```

**Step 3: Add CI step**

```yaml
# In CI workflow
- name: No Bug-Hiding Check
  run: uv run python scripts/cicd/no_bug_hiding.py check --root src/elspeth
```

**Step 4: Commit**

```bash
git add config/cicd/no_bug_hiding.yaml .github/workflows/ci.yaml
git commit -m "chore(ci): add no-bug-hiding enforcement with burn-down allowlist

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Architectural Bug Fixes (P1 Bugs)

These bugs represent integration seam mismatches that must be fixed before defensive patterns can be properly remediated.

### Task 2.1: Fix Source/Sink Lifecycle Hooks

**Bug:** Orchestrator only calls lifecycle hooks on transforms, not sources/sinks. Additionally, `hasattr` checks are used even though base classes define these methods.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:303-410`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write failing test for source lifecycle**

```python
# tests/engine/test_orchestrator.py
def test_source_lifecycle_hooks_called(tmp_path, mocker):
    """Source on_start and on_complete should be called."""
    on_start_spy = mocker.spy(CSVSource, 'on_start')
    on_complete_spy = mocker.spy(CSVSource, 'on_complete')

    # ... setup config with CSV source ...
    orchestrator = Orchestrator(config)
    orchestrator.run()

    on_start_spy.assert_called_once()
    on_complete_spy.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_orchestrator.py::test_source_lifecycle_hooks_called -v`
Expected: FAIL

**Step 3: Implement source lifecycle calls in orchestrator**

In `_execute_run()`, add before source.load():
```python
# Call source on_start - no hasattr needed, base class provides no-op
source.on_start(source_ctx)
```

And in finally block:
```python
# Call source on_complete - no hasattr needed
source.on_complete(source_ctx)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_orchestrator.py::test_source_lifecycle_hooks_called -v`
Expected: PASS

**Step 5: Write failing test for sink lifecycle**

```python
def test_sink_lifecycle_hooks_called(tmp_path, mocker):
    """Sink on_start and on_complete should be called."""
    # Similar pattern for sinks
```

**Step 6-8: Implement and verify sink lifecycle**

**Step 9: Remove hasattr checks from transform lifecycle**

```python
# Before:
if hasattr(transform, "on_start"):
    transform.on_start(ctx)

# After - base class always provides these methods:
transform.on_start(ctx)
```

**Step 10: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "fix(engine): call lifecycle hooks on all plugins, remove hasattr checks

Fixes: source/sink on_start and on_complete were never called
Fixes: hasattr checks were defensive when base class provides methods
Impact: Plugins can now properly initialize and cleanup resources

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Fix Run Configuration Recording

**Bug:** `PipelineConfig.config` is never populated, so runs store `{}`.

**Files:**
- Modify: `src/elspeth/cli.py:306`
- Modify: `src/elspeth/core/config.py` (add resolver)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write failing test**

```python
def test_run_records_resolved_config(tmp_path, landscape_db):
    """Run should record the full resolved configuration."""
    # Create a pipeline with known config
    config = PipelineConfig(
        source=SourceConfig(plugin="csv", options={"path": "data.csv"}),
        transforms=[],
        sinks=[SinkConfig(name="out", plugin="csv", options={"path": "out.csv"})],
        config={"source": {"plugin": "csv"}, "sinks": [{"name": "out"}]}  # Resolved config
    )

    orchestrator = Orchestrator(config, db_path=landscape_db)
    run = orchestrator.run()

    # Verify config was recorded
    recorder = LandscapeRecorder(landscape_db)
    stored_run = recorder.get_run(run.run_id)
    assert stored_run.settings_json != {}
    assert "source" in stored_run.settings_json
```

**Step 2: Run test to verify it fails**

**Step 3: Add config resolver utility**

```python
# src/elspeth/core/config.py
def resolve_pipeline_config(settings_path: Path) -> tuple[dict, str]:
    """Resolve and canonicalize pipeline configuration.

    Returns:
        (resolved_config_dict, config_hash)
    """
    # Load and merge config sources
    # Apply secret fingerprinting
    # Return canonical form
```

**Step 4: Update CLI to use resolver**

```python
# src/elspeth/cli.py in _execute_pipeline
resolved_config, config_hash = resolve_pipeline_config(settings_path)
pipeline_config = PipelineConfig(
    source=source_config,
    transforms=transform_configs,
    sinks=sink_configs,
    config=resolved_config,  # Now populated!
)
```

**Step 5-6: Run tests, verify pass**

**Step 7: Commit**

---

### Task 2.3: Fix Node Metadata from Plugin Classes

**Bug:** All nodes register with hardcoded `plugin_version="1.0.0"`.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:243`
- Modify: `src/elspeth/plugins/manager.py` (add metadata accessor)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write failing test**

```python
def test_node_metadata_from_plugin(tmp_path, landscape_db):
    """Node registration should use actual plugin metadata."""
    # Create plugin with explicit version
    class CustomTransform(BaseTransform):
        name = "custom"
        plugin_version = "2.5.0"
        determinism = Determinism.NONDETERMINISTIC

        def process(self, row, ctx):
            return TransformResult.ok(row)

    # Register and run
    # ...

    # Verify node has correct metadata
    node = recorder.get_node(run.run_id, "custom")
    assert node.plugin_version == "2.5.0"
    assert node.determinism == "nondeterministic"
```

**Step 2-6: Implement and verify**

**Step 7: Commit**

---

### Task 2.4: Fix Non-File Sink Artifact Registration

**Bug:** Database sinks crash with KeyError when registering artifacts.

**Files:**
- Create: `src/elspeth/engine/artifacts.py` (ArtifactDescriptor dataclass)
- Modify: `src/elspeth/engine/adapters.py`
- Modify: `src/elspeth/engine/executors.py:780`
- Modify: `src/elspeth/core/landscape/recorder.py` (accept ArtifactDescriptor)
- Modify: `src/elspeth/core/landscape/schema.py` (path → path_or_uri)
- Test: `tests/engine/test_executors.py`

**Step 1: Write failing test**

```python
def test_database_sink_artifact_registration(tmp_path, landscape_db):
    """Database sink should register artifact without crashing."""
    config = PipelineConfig(
        source=SourceConfig(plugin="csv", options={"path": str(tmp_path / "in.csv")}),
        transforms=[],
        sinks=[SinkConfig(
            name="db_out",
            plugin="database",
            options={"url": "sqlite:///test.db", "table": "output"}
        )],
    )

    # Should not raise KeyError
    orchestrator = Orchestrator(config, db_path=landscape_db)
    run = orchestrator.run()

    # Verify artifact was recorded
    artifacts = recorder.get_artifacts(run.run_id)
    assert len(artifacts) == 1
    assert "sqlite" in artifacts[0].path_or_uri
```

**Step 2: Run test, verify KeyError**

**Step 3: Create ArtifactDescriptor dataclass**

```python
# src/elspeth/engine/artifacts.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ArtifactDescriptor:
    """Unified artifact descriptor for all sink types.

    Provides canonical URI schemes for different artifact types,
    ensuring consistent audit trail recording regardless of sink.
    """
    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str
    content_hash: str
    size_bytes: int
    metadata: dict | None = None

    @classmethod
    def for_file(cls, path: str, content_hash: str, size_bytes: int) -> "ArtifactDescriptor":
        """Create descriptor for file-based artifacts."""
        return cls("file", f"file://{path}", content_hash, size_bytes)

    @classmethod
    def for_database(
        cls, url: str, table: str, content_hash: str, payload_size: int, row_count: int
    ) -> "ArtifactDescriptor":
        """Create descriptor for database artifacts.

        Args:
            url: Database connection URL
            table: Target table name
            content_hash: Hash of canonical JSON payload written
            payload_size: Size in bytes of the serialized payload
            row_count: Number of rows written (stored in metadata)
        """
        return cls(
            "database",
            f"db://{table}@{url}",
            content_hash,
            payload_size,
            {"table": table, "row_count": row_count}
        )

    @classmethod
    def for_webhook(
        cls, url: str, content_hash: str, request_size: int, response_code: int
    ) -> "ArtifactDescriptor":
        """Create descriptor for webhook artifacts."""
        return cls(
            "webhook",
            f"webhook://{url}",
            content_hash,
            request_size,
            {"response_code": response_code}
        )
```

**Step 4: Update Landscape schema (migration)**

```python
# src/elspeth/core/landscape/schema.py
# Rename 'path' column to 'path_or_uri' to support URI schemes
artifacts = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String, primary_key=True),
    Column("run_id", String, ForeignKey("runs.run_id"), nullable=False),
    Column("produced_by_state_id", String, ForeignKey("node_states.state_id"), nullable=False),
    Column("sink_node_id", String, ForeignKey("nodes.node_id"), nullable=False),
    Column("artifact_type", String, nullable=False),
    Column("path_or_uri", String, nullable=False),  # Changed from 'path'
    Column("content_hash", String, nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("metadata_json", String),  # New: stores ArtifactDescriptor.metadata
    Column("created_at", DateTime, nullable=False),
)
```

**Step 5: Update LandscapeRecorder.register_artifact()**

```python
def register_artifact(
    self,
    run_id: str,
    state_id: str,
    sink_node_id: str,
    descriptor: ArtifactDescriptor,
) -> str:
    """Register an artifact using unified descriptor."""
    artifact_id = str(uuid.uuid4())
    # ... insert using descriptor fields
```

**Step 6: Update adapters and executors to use ArtifactDescriptor**

**Step 7: Commit**

---

## Phase 3: Defensive Pattern Remediation

### Task 3.0: Create PluginConfig Base Infrastructure

**Rationale:** Before converting individual plugins, create shared infrastructure for typed configs.

**Files:**
- Create: `src/elspeth/plugins/config_base.py`
- Test: `tests/plugins/test_config_base.py`

**Step 1: Create base config class**

```python
# src/elspeth/plugins/config_base.py
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import Any

class PluginConfig(BaseModel):
    """Base class for typed plugin configurations.

    Provides common validation patterns and helpful error messages.
    All plugin configs should inherit from this class.
    """

    model_config = {"extra": "forbid"}  # Reject unknown fields

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PluginConfig":
        """Create config from dict with clear error on validation failure."""
        try:
            return cls(**config)
        except ValidationError as e:
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {e}"
            ) from e


class PathConfig(PluginConfig):
    """Base for configs that include file paths."""
    path: str

    @field_validator("path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("path cannot be empty")
        return v

    def resolved_path(self, base_dir: Path | None = None) -> Path:
        """Resolve path relative to base directory if provided."""
        p = Path(self.path)
        if base_dir and not p.is_absolute():
            return base_dir / p
        return p
```

**Step 2: Write tests**

```python
def test_plugin_config_rejects_extra_fields():
    """Extra fields should raise validation error."""
    class MyConfig(PluginConfig):
        name: str

    with pytest.raises(ValidationError):
        MyConfig(name="test", unknown_field="value")

def test_path_config_rejects_empty():
    """Empty path should raise validation error."""
    class FileConfig(PathConfig):
        pass

    with pytest.raises(ValidationError):
        FileConfig(path="")
```

**Step 3: Commit**

---

### Task 3.1: Categorize All Violations

Before fixing, categorize each violation:

| Category | Action | Example |
|----------|--------|---------|
| **Trust Boundary** | Whitelist with justification | TUI displaying external data |
| **Optional API** | Whitelist as intentional design | `PluginManager.get_*_by_name()` |
| **Cleanup Isolation** | Whitelist with justification | `suppress(Exception)` in cleanup |
| **Plugin Config** | Replace with typed Pydantic config | `config.get("path")` → `config.path` |
| **Internal Invariant** | Direct access + fix caller | `self._map.get(key)` → `self._map[key]` |
| **Introspection** | Replace with protocol/enum | `hasattr(plugin, "on_start")` |

**Step 1: Run categorization script**

```bash
uv run python -c "
import json
with open('/tmp/violations.json') as f:
    d = json.load(f)

# Categorize by file pattern and context
trust_boundary = []     # TUI, display code
optional_api = []       # PluginManager lookups
cleanup_isolation = []  # suppress(Exception)
plugin_config = []      # Sources, sinks, transforms config
internal = []           # Engine, core
introspection = []      # hasattr checks

for v in d['violations']:
    if 'tui/' in v['file']:
        trust_boundary.append(v)
    elif 'manager.py' in v['file'] and 'get_' in v['context'][0]:
        optional_api.append(v)
    elif 'suppress' in v.get('code', ''):
        cleanup_isolation.append(v)
    elif v['rule_id'] == 'R3':
        introspection.append(v)
    elif 'plugins/' in v['file'] and '.get(' in v['code']:
        plugin_config.append(v)
    else:
        internal.append(v)

print(f'Trust boundary (whitelist): {len(trust_boundary)}')
print(f'Optional API (whitelist): {len(optional_api)}')
print(f'Cleanup isolation (whitelist): {len(cleanup_isolation)}')
print(f'Plugin config (typed config): {len(plugin_config)}')
print(f'Internal (direct access): {len(internal)}')
print(f'Introspection (protocols): {len(introspection)}')
"
```

---

### Task 3.2: Whitelist Legitimate Patterns

**Files:**
- Modify: `config/cicd/no_bug_hiding.yaml`

**Step 1: Add TUI trust boundary entries**

```yaml
# TUI Display Code - Trust Boundary
# These display data from Landscape database which may have missing fields
# for incomplete/failed runs. Graceful degradation is correct behavior.

  - key: "tui/widgets/node_detail.py:R1:NodeDetailWidget:compose:line=38"
    owner: "architecture"
    reason: "Trust boundary: displaying Landscape data that may be incomplete for failed runs"
    safety: "Graceful degradation to 'unknown' - user sees partial data rather than crash"
    expires: null  # Permanent - intentional design
```

**Step 2: Add PluginManager Optional API entries**

```yaml
# Plugin Manager - Intentional Optional API
# These methods return Optional[T] by design - callers handle None case.
# NOT defensive coding - this is the documented API contract.

  - key: "plugins/manager.py:R1:PluginManager:get_source_by_name:line=245"
    owner: "architecture"
    reason: "Intentional Optional API: returns None for unknown plugin names"
    safety: "Callers check return value - this is the documented contract"
    expires: null  # Permanent - intentional design
```

**Step 3: Add cleanup isolation entry**

```yaml
# Orchestrator Cleanup - Isolation Pattern
# One plugin's cleanup failure should not prevent other plugins from cleanup.

  - key: "engine/orchestrator.py:R4:Orchestrator:_cleanup:line=404"
    owner: "architecture"
    reason: "Cleanup isolation: one plugin failure must not prevent others from cleanup"
    safety: "Errors are logged; cleanup continues for remaining plugins"
    expires: null  # Permanent - intentional design
```

**Step 4: Verify allowlist matches**

```bash
uv run python scripts/cicd/no_bug_hiding.py check --root src/elspeth
# Should show reduced violation count
```

**Step 5: Commit**

---

### Task 3.3: Replace Plugin Config dict.get with Typed Configs

**Files:**
- Modify: `src/elspeth/plugins/sources/csv_source.py`
- Modify: `src/elspeth/plugins/sources/json_source.py`
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Modify: `src/elspeth/plugins/sinks/database_sink.py`
- Modify: `src/elspeth/plugins/transforms/field_mapper.py`
- Modify: `src/elspeth/plugins/gates/*.py`

**Step 1: Write failing test for CSVSource**

```python
def test_csv_source_rejects_invalid_config():
    """CSVSource should fail fast on invalid config, not silently default."""
    with pytest.raises(PluginConfigError):
        CSVSource({})  # Missing required 'path'
```

**Step 2: Create typed config for CSVSource**

```python
# src/elspeth/plugins/sources/csv_source.py
from elspeth.plugins.config_base import PathConfig

class CSVSourceConfig(PathConfig):
    """Typed configuration for CSV source."""
    delimiter: str = ","
    encoding: str = "utf-8"
    skip_rows: int = 0

class CSVSource(BaseSource):
    name = "csv"

    def __init__(self, config: dict[str, Any]) -> None:
        # Validate and parse config - fails fast on invalid
        self._config = CSVSourceConfig.from_dict(config)
        self._path = self._config.resolved_path()
        self._delimiter = self._config.delimiter
        # No more .get() calls!
```

**Step 3: Run tests, verify pass**

**Step 4: Repeat for each plugin (one commit per plugin or per category)**

---

### Task 3.4: Fix Internal dict.get Usage

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `src/elspeth/engine/adapters.py`
- Modify: `src/elspeth/core/dag.py`

**Note:** PluginManager `.get()` methods are intentional Optional API and should NOT be changed.

**Step 1: Analyze each internal .get() usage**

For each, determine:
1. Can the key be missing? If yes, why?
2. Should we fail fast or is there a valid default?

**Step 2: Fix GateExecutor._route_resolution_map.get()**

```python
# Before:
destination = self._route_resolution_map.get(label, "continue")

# After - the map must contain all labels from config
destination = self._route_resolution_map[label]  # KeyError = config bug
```

**Step 3: Add validation at construction time**

```python
def __init__(self, ..., route_resolution_map: dict[str, str]):
    # Validate all expected labels are present
    expected_labels = self._get_expected_labels()
    missing = expected_labels - route_resolution_map.keys()
    if missing:
        raise ConfigurationError(f"Missing route resolution for labels: {missing}")
    self._route_resolution_map = route_resolution_map
```

**Step 4: Repeat for each internal .get() usage**

---

### Task 3.5: Fix Broad Exception Handlers

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Modify: `src/elspeth/core/landscape/recorder.py`

**Note:** Orchestrator cleanup `suppress(Exception)` is whitelisted in Task 3.2.

**Step 1: Identify R4 violations (excluding whitelisted)**

```python
# explain_screen.py:81 - catches Exception and returns None
# explain_screen.py:143 - catches Exception and returns None
# recorder.py:1665 - catches multiple exceptions and passes
```

**Step 2: Fix explain_screen.py**

```python
# Before:
try:
    self._lineage_data = self._load_lineage()
except Exception:
    self._lineage_data = None

# After - catch specific exceptions
try:
    self._lineage_data = self._load_lineage()
except (DatabaseError, LineageNotFoundError) as e:
    self._lineage_data = None
    self._error_message = str(e)  # Display error to user
```

**Step 3: Fix recorder.py payload retrieval**

```python
# Before:
try:
    content = payload_store.get(content_hash)
    return json.loads(content)
except (KeyError, json.JSONDecodeError, OSError):
    pass  # Silent failure!

# After - explicit handling
try:
    content = payload_store.get(content_hash)
    return json.loads(content)
except KeyError:
    raise PayloadNotFoundError(f"Payload {content_hash} has been purged")
except json.JSONDecodeError as e:
    raise PayloadCorruptedError(f"Payload {content_hash} is corrupted: {e}")
except OSError as e:
    raise PayloadAccessError(f"Cannot access payload {content_hash}: {e}")
```

**Step 4: Commit**

---

## Phase 4: Verification and CI Lock-Down

### Task 4.1: Verify Zero Violations (Excluding Whitelist)

**Step 1: Run final check**

```bash
uv run python scripts/cicd/no_bug_hiding.py check --root src/elspeth
```

Expected: `No bug-hiding patterns detected. Check passed.`

**Step 2: Remove burn-down expiry dates from legitimate whitelist entries**

For permanent trust-boundary exceptions, set `expires: null`.

**Step 3: Commit final allowlist**

---

### Task 4.2: Add Pre-Commit Hook

**Files:**
- Modify: `.pre-commit-config.yaml`

```yaml
- repo: local
  hooks:
    - id: no-bug-hiding
      name: No Bug-Hiding Patterns
      entry: uv run python scripts/cicd/no_bug_hiding.py check --root src/elspeth
      language: system
      types: [python]
      pass_filenames: false
```

---

### Task 4.3: Add Property Tests for TUI Degradation

**Files:**
- Create: `tests/tui/test_graceful_degradation.py`

```python
from hypothesis import given, strategies as st

@given(st.dictionaries(st.text(), st.none() | st.text() | st.integers()))
def test_node_detail_handles_incomplete_data(incomplete_data):
    """NodeDetailWidget should not crash on incomplete/malformed data."""
    widget = NodeDetailWidget(incomplete_data)
    # Should render without raising
    widget.compose()
```

---

## Success Criteria

1. `no_bug_hiding.py check` passes in CI
2. All 5 P1 bugs closed
3. Zero violations outside whitelist
4. All whitelist entries have:
   - Explicit owner
   - Clear justification (trust boundary / optional API / cleanup isolation)
   - Safety explanation
   - Expiry date (or `null` for permanent)
5. Pre-commit hook prevents regressions
6. Property tests verify TUI graceful degradation

---

## Estimated Effort

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | 1 | 1 hour |
| Phase 2 | 4 | 6-8 hours |
| Phase 3 | 5 | 8-10 hours |
| Phase 4 | 3 | 1 hour |
| **Total** | **13** | **~16-20 hours** |

**Revised from original 12 hours based on:**
- Schema migration complexity for artifacts table
- PluginConfig base class infrastructure
- Property test additions
- More careful categorization work

---

## Risk Mitigation

1. **Breaking Changes:** Each task is atomic and committed separately. Easy rollback.
2. **Test Coverage:** Every fix includes a failing test first (TDD).
3. **Gradual Rollout:** Allowlist enables incremental progress - CI doesn't break.
4. **Owner Accountability:** Each whitelist entry has an owner who is responsible.
5. **Schema Migration:** Artifacts table change is additive (rename column) - existing data preserved.
6. **API Stability:** PluginManager Optional API preserved - no breaking changes to callers.
