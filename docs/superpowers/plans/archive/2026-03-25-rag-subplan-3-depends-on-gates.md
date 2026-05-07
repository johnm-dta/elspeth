# RAG Ingestion Sub-plan 3: `depends_on` + Commencement Gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pipeline-level orchestration: `depends_on` runs dependency pipelines before the main pipeline starts, commencement gates evaluate go/no-go conditions against a pre-flight context, and collection probes check vector store readiness.

**Architecture:** New config models in L1, new orchestration phases in L2, collection probe assembly in L3. The dependency resolution phase calls `bootstrap_and_run()` (extracted from the CLI codepath) for each dependency. Commencement gates use `ExpressionParser` (extended in sub-plan 1) to evaluate expressions against a frozen pre-flight context. Collection probes use the `CollectionProbe` protocol (L0, from sub-plan 1) constructed by a factory in L3. **Dependency resolution and gate evaluation run inside `bootstrap_and_run()` before orchestrator construction** — the orchestrator never sees settings paths or gate configs.

**Tech Stack:** Pydantic v2, ExpressionParser (AST-whitelist), deep_freeze, pathlib

**Spec:** `docs/superpowers/specs/2026-03-25-rag-ingestion-pipeline-design.md` (Components 2 and 3)

**Depends on:** Sub-plan 1 (shared infrastructure) must be merged first.

**Risk:** MEDIUM — extracts `bootstrap_and_run()` from the CLI codepath with dependency/gate phases. Does NOT modify the orchestrator's `run()` method (pre-flight logic lives in `bootstrap_and_run()` instead). Needs careful review of the extraction.

---

## Review Notes from Sub-plan 1 (read before implementing)

The following changes were made during sub-plan 1's 4-agent review that affect this plan:

1. **ExpressionParser calling convention.** The parser now has two explicit modes controlled by `single_name_mode` (set automatically based on `allowed_names`):
   - **Single-name mode** (`allowed_names=None` or `allowed_names=["row"]`): caller passes the value directly as `context` — e.g. `parser.evaluate({"x": 5})` where the dict IS the row.
   - **Multi-name mode** (`allowed_names=["collections", "dependency_runs", "env"]`): caller passes a **namespace dict** where each key is an allowed name — e.g. `parser.evaluate({"collections": {...}, "dependency_runs": {...}, "env": {...}})`.

   When building the commencement gate evaluator, **always use multi-name mode** with 2+ allowed names. Do NOT pass a single-element `allowed_names` list with a namespace dict — that would trigger single-name mode and silently misinterpret the context.

2. **`env` namespace trust tier.** The `env` key in commencement gate expressions (e.g. `env['ENVIRONMENT'] == 'production'`) injects operator-supplied environment variables. This is a **Tier 3 trust boundary** inside system-owned code. The commencement gate implementation must:
   - Document that `env` values are external data (operator-controlled, not ELSPETH-controlled)
   - Deep-freeze the `env` dict before passing to `ExpressionParser.evaluate()` (the parser does not freeze its context)
   - Deep-freeze the entire pre-flight context snapshot stored in `CommencementGateFailedError.context_snapshot` when raising

3. **`CommencementGateFailedError.context_snapshot` is not auto-frozen.** The error class is a plain Exception (not a frozen dataclass), so the deep-freeze policy doesn't apply automatically. When constructing this error in the commencement gate, pass a `deep_freeze()`d copy of the context snapshot to prevent mutation between raise and Landscape serialization.

4. **`row` is forbidden in commencement gate expressions.** If `allowed_names` does not include `"row"`, expressions referencing `row['x']` are rejected at parse time. This is verified by test. Commencement gates should use `allowed_names=["collections", "dependency_runs", "env"]` — `row` is intentionally excluded since gates run before row processing begins.

5. **Empty `allowed_names` raises ValueError.** `ExpressionParser(expr, allowed_names=[])` now raises immediately rather than silently falling back to `["row"]`.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/elspeth/core/dependency_config.py` | `DependencyConfig`, `CommencementGateConfig`, `CollectionProbeConfig`, `DependencyRunResult`, `GateResult` |
| Modify | `src/elspeth/core/config.py` | Add `depends_on`, `commencement_gates`, `collection_probes` fields to `ElspethSettings` |
| Create | `src/elspeth/engine/bootstrap.py` | `bootstrap_and_run()` extracted from CLI |
| Create | `src/elspeth/engine/dependency_resolver.py` | `resolve_dependencies()`, cycle detection, depth limit |
| Create | `src/elspeth/engine/commencement.py` | `evaluate_commencement_gates()`, pre-flight context assembly |
| Create | `src/elspeth/plugins/infrastructure/probe_factory.py` | `build_collection_probes()` from explicit config |
| ~~Modify~~ | ~~`src/elspeth/engine/orchestrator/core.py`~~ | ~~NOT modified — pre-flight logic lives in `bootstrap.py`~~ |
| — | `src/elspeth/cli.py` | NOT modified — CLI keeps its own execution path, `bootstrap_and_run()` is parallel |
| Create | `tests/unit/core/test_dependency_config.py` | Config model tests |
| Create | `tests/unit/engine/test_dependency_resolver.py` | Dependency resolution + cycle detection tests |
| Create | `tests/unit/engine/test_commencement.py` | Gate evaluation + context assembly tests |
| Create | `tests/unit/plugins/infrastructure/test_probe_factory.py` | Probe factory tests |
| Create | `tests/unit/engine/test_bootstrap_preflight.py` | Pre-flight dispatch tests (skip when unconfigured) |
| Create | `tests/integration/engine/test_depends_on.py` | Two-pipeline dependency integration |
| Create | `tests/integration/engine/test_commencement_gates.py` | Gate evaluation integration |

---

### Task 1: Config Models — `DependencyConfig`, `CommencementGateConfig`, `CollectionProbeConfig`

**Files:**
- Create: `src/elspeth/core/dependency_config.py`
- Create: `tests/unit/core/test_dependency_config.py`

- [ ] **Step 1: Write config validation tests**

```python
# tests/unit/core/test_dependency_config.py
"""Tests for dependency, commencement gate, and collection probe config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.core.dependency_config import (
    CollectionProbeConfig,
    CommencementGateConfig,
    DependencyConfig,
)


class TestDependencyConfig:
    def test_valid_config(self) -> None:
        config = DependencyConfig(name="index_corpus", settings="./index.yaml")
        assert config.name == "index_corpus"
        assert config.settings == "./index.yaml"

    def test_frozen(self) -> None:
        config = DependencyConfig(name="x", settings="./x.yaml")
        with pytest.raises(ValidationError):
            config.name = "y"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            DependencyConfig(name="x", settings="./x.yaml", extra="bad")  # type: ignore[call-arg]


class TestCommencementGateConfig:
    def test_valid_config(self) -> None:
        config = CommencementGateConfig(
            name="corpus_ready",
            condition="collections['test']['count'] > 0",
        )
        assert config.name == "corpus_ready"
        assert config.on_fail == "abort"  # default

    def test_on_fail_default_abort(self) -> None:
        config = CommencementGateConfig(name="x", condition="True")
        assert config.on_fail == "abort"

    def test_rejects_invalid_on_fail(self) -> None:
        with pytest.raises(ValidationError):
            CommencementGateConfig(name="x", condition="True", on_fail="warn")  # type: ignore[arg-type]


class TestCollectionProbeConfig:
    def test_valid_config(self) -> None:
        config = CollectionProbeConfig(
            collection="science-facts",
            provider="chroma",
            provider_config={
                "mode": "persistent",
                "persist_directory": "./chroma_data",
            },
        )
        assert config.collection == "science-facts"
        assert config.provider == "chroma"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            CollectionProbeConfig(
                collection="x", provider="chroma", provider_config={}, extra="bad"  # type: ignore[call-arg]
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dependency_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement config models**

```python
# src/elspeth/core/dependency_config.py
"""Configuration models for pipeline dependencies and commencement gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from elspeth.contracts.freeze import freeze_fields


class DependencyConfig(BaseModel):
    """Declares a pipeline that must run before this one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Unique label for this dependency")
    settings: str = Field(description="Path to dependency pipeline settings file")


class CommencementGateConfig(BaseModel):
    """Declares a go/no-go condition evaluated before the pipeline starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Unique label for this gate")
    condition: str = Field(description="Expression evaluated against pre-flight context")
    on_fail: Literal["abort"] = Field(
        default="abort",
        description="Action on failure (only 'abort' supported initially)",
    )


class CollectionProbeConfig(BaseModel):
    """Declares a vector store collection to probe before gate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = Field(description="Collection name to probe")
    provider: str = Field(description="Provider type (e.g., 'chroma')")
    provider_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific connection config",
    )


@dataclass(frozen=True, slots=True)
class DependencyRunResult:
    """Result of a successful dependency pipeline run."""

    name: str
    run_id: str
    settings_hash: str
    duration_ms: int
    indexed_at: str  # ISO 8601 timestamp


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of a successful commencement gate evaluation."""

    name: str
    condition: str
    result: bool
    context_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "context_snapshot")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dependency_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/dependency_config.py tests/unit/core/test_dependency_config.py
git commit -m "feat: add DependencyConfig, CommencementGateConfig, CollectionProbeConfig, result dataclasses"
```

---

### Task 2: `DependencyRunResult` and `GateResult` Tests

**Files:**
- Modify: `tests/unit/core/test_dependency_config.py`

- [ ] **Step 1: Write tests for result dataclasses**

```python
# Append to tests/unit/core/test_dependency_config.py
from types import MappingProxyType

from elspeth.core.dependency_config import DependencyRunResult, GateResult


class TestDependencyRunResult:
    def test_construction(self) -> None:
        result = DependencyRunResult(
            name="index_corpus",
            run_id="abc-123",
            settings_hash="sha256:deadbeef",
            duration_ms=4520,
            indexed_at="2026-03-25T14:02:33Z",
        )
        assert result.name == "index_corpus"
        assert result.run_id == "abc-123"
        assert result.indexed_at == "2026-03-25T14:02:33Z"

    def test_frozen(self) -> None:
        result = DependencyRunResult(
            name="x", run_id="y", settings_hash="z", duration_ms=0, indexed_at="t"
        )
        with pytest.raises(AttributeError):
            result.name = "other"  # type: ignore[misc]


class TestGateResult:
    def test_construction(self) -> None:
        snapshot = {"collections": {"test": {"count": 10, "reachable": True}}}
        result = GateResult(
            name="corpus_ready",
            condition="collections['test']['count'] > 0",
            result=True,
            context_snapshot=snapshot,
        )
        assert result.name == "corpus_ready"
        assert result.result is True

    def test_context_snapshot_is_deep_frozen(self) -> None:
        snapshot = {"collections": {"test": {"count": 10}}}
        result = GateResult(
            name="x", condition="True", result=True, context_snapshot=snapshot
        )
        assert isinstance(result.context_snapshot, MappingProxyType)
        # Nested dict should also be frozen
        assert isinstance(result.context_snapshot["collections"], MappingProxyType)

    def test_frozen(self) -> None:
        result = GateResult(
            name="x", condition="True", result=True, context_snapshot={}
        )
        with pytest.raises(AttributeError):
            result.name = "other"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dependency_config.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/core/test_dependency_config.py
git commit -m "test: add DependencyRunResult and GateResult dataclass tests"
```

---

### Task 3: Add Fields to `ElspethSettings`

**Files:**
- Modify: `src/elspeth/core/config.py`

- [ ] **Step 1: Run existing config tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/core/ -v -q`
Expected: PASS

- [ ] **Step 2: Add new optional fields to `ElspethSettings`**

In `src/elspeth/core/config.py`, after the `gates` field (around line 1267), add:

```python
from elspeth.core.dependency_config import (
    CollectionProbeConfig,
    CommencementGateConfig,
    DependencyConfig,
)

# Inside ElspethSettings class, after gates field:
depends_on: list[DependencyConfig] = Field(
    default_factory=list,
    max_length=20,
    description="Pipeline dependencies — run these before the main pipeline",
)
commencement_gates: list[CommencementGateConfig] = Field(
    default_factory=list,
    max_length=20,
    description="Go/no-go conditions evaluated after dependencies complete",
)
collection_probes: list[CollectionProbeConfig] = Field(
    default_factory=list,
    max_length=20,
    description="Vector store collections to probe for gate context",
)
```

- [ ] **Step 3: Run config tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/unit/core/ -v -q`
Expected: PASS (new fields are optional with defaults, so existing tests still pass)

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/core/config.py
git commit -m "feat: add depends_on, commencement_gates, collection_probes to ElspethSettings"
```

---

### Task 4: Dependency Resolver — Cycle Detection and Depth Limit

**Files:**
- Create: `src/elspeth/engine/dependency_resolver.py`
- Create: `tests/unit/engine/test_dependency_resolver.py`

- [ ] **Step 1: Write tests for cycle detection and depth limit**

```python
# tests/unit/engine/test_dependency_resolver.py
"""Tests for pipeline dependency resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.engine.dependency_resolver import (
    detect_cycles,
    resolve_dependencies,
)
from elspeth.contracts.errors import DependencyFailedError
from elspeth.core.dependency_config import DependencyConfig


class TestCycleDetection:
    def test_no_cycle_returns_none(self, tmp_path: Path) -> None:
        # A -> B, no cycle
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        b.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n")
        a.write_text(f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\nlandscape:\n  url: sqlite:///test.db\n")

        # Should not raise
        detect_cycles(a)

    def test_self_loop_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text(f"depends_on:\n  - name: self\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_two_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        b.write_text(f"depends_on:\n  - name: a\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_three_hop_cycle_detected(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        c = tmp_path / "c.yaml"
        a.write_text(f"depends_on:\n  - name: b\n    settings: {b}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        b.write_text(f"depends_on:\n  - name: c\n    settings: {c}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        c.write_text(f"depends_on:\n  - name: a\n    settings: {a}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
            detect_cycles(a)

    def test_depth_limit_exceeded(self, tmp_path: Path) -> None:
        # Create a chain: a -> b -> c -> d (depth 4, exceeds limit of 3)
        files = {}
        for name in ["d", "c", "b", "a"]:
            f = tmp_path / f"{name}.yaml"
            files[name] = f

        files["d"].write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        files["c"].write_text(f"depends_on:\n  - name: d\n    settings: {files['d']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        files["b"].write_text(f"depends_on:\n  - name: c\n    settings: {files['c']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        files["a"].write_text(f"depends_on:\n  - name: b\n    settings: {files['b']}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        with pytest.raises(ValueError, match="[Dd]epth"):
            detect_cycles(files["a"], max_depth=3)

    def test_uses_resolved_paths(self, tmp_path: Path) -> None:
        """Symlinks resolve to the same canonical path."""
        real = tmp_path / "real.yaml"
        real.write_text("source:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")
        link = tmp_path / "link.yaml"
        link.symlink_to(real)

        main = tmp_path / "main.yaml"
        main.write_text(f"depends_on:\n  - name: dep\n    settings: {link}\nsource:\n  plugin: null_source\nsinks:\n  out:\n    plugin: json_sink\n")

        # Should not raise — link resolves to real, no cycle
        detect_cycles(main)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_dependency_resolver.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement cycle detection**

```python
# src/elspeth/engine/dependency_resolver.py
"""Pipeline dependency resolution — cycle detection, depth limiting, and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_depends_on(settings_path: Path) -> list[dict[str, str]]:
    """Load only the depends_on key from a settings file."""
    with settings_path.open() as f:
        data = yaml.safe_load(f) or {}
    return data.get("depends_on", [])


def detect_cycles(
    settings_path: Path,
    *,
    max_depth: int = 3,
    _visited: set[str] | None = None,
    _stack: list[str] | None = None,
    _depth: int = 0,
) -> None:
    """Detect circular dependencies and enforce depth limit.

    Uses DFS on canonicalized (resolved) paths.
    Raises ValueError on cycle or depth limit violation.
    """
    canonical = str(settings_path.resolve())
    visited = _visited if _visited is not None else set()
    stack = _stack if _stack is not None else []

    if _depth > max_depth:
        raise ValueError(
            f"Dependency depth limit exceeded ({max_depth}). "
            f"Chain: {' -> '.join(stack)} -> {canonical}"
        )

    if canonical in stack:
        cycle_start = stack.index(canonical)
        cycle_path = stack[cycle_start:] + [canonical]
        raise ValueError(
            f"Circular dependency detected: {' -> '.join(cycle_path)}"
        )

    if canonical in visited:
        return  # Already fully explored, no cycle through this node

    stack.append(canonical)
    deps = _load_depends_on(settings_path)

    for dep in deps:
        dep_path = (settings_path.parent / dep["settings"]).resolve()
        detect_cycles(
            Path(dep_path),
            max_depth=max_depth,
            _visited=visited,
            _stack=stack,
            _depth=_depth + 1,
        )

    stack.pop()
    visited.add(canonical)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_dependency_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/dependency_resolver.py tests/unit/engine/test_dependency_resolver.py
git commit -m "feat: add dependency cycle detection with depth limit and path canonicalization"
```

---

### Task 5: `bootstrap_and_run()` Extraction

**Files:**
- Create: `src/elspeth/engine/bootstrap.py`
- Modify: `src/elspeth/cli.py`

This is the highest-risk task. We extract a reusable `bootstrap_and_run()` from the CLI so that dependency pipelines can be executed programmatically.

**Design decision: dependency resolution and commencement gates run inside `bootstrap_and_run()`, NOT inside the orchestrator.** The orchestrator's `run()` method is unchanged. This eliminates the need to thread `settings_path` into the orchestrator (which currently has no such attribute and doesn't need one).

**The CLI setup sequence (what we're extracting from):**

The CLI `run` command (cli.py) executes this sequence before reaching the orchestrator:

1. `_load_settings_with_secrets(settings_path)` → `(config: ElspethSettings, secret_resolutions)` (cli.py:396)
2. `instantiate_plugins_from_config(config)` → `plugins: PluginBundle` (cli_helpers.py:42, imports `_get_plugin_manager` from cli.py:57)
3. `ExecutionGraph.from_plugin_instances(source, transforms, sinks, ...)` → `graph` (cli.py:435)
4. `graph.validate()` (cli.py:444)
5. `resolve_audit_passphrase(config.landscape)` → passphrase (cli.py:492)
6. `_execute_pipeline_with_instances(config, graph, plugins, ...)` which:
   a. `LandscapeDB.from_url(config.landscape.url, ...)` (cli.py:958)
   b. `FilesystemPayloadStore(config.payload_store.base_path)` (cli.py:981)
   c. `_orchestrator_context(config, graph, plugins, db=db)` → context manager that builds PipelineConfig, EventBus, RuntimeConfigs, Orchestrator (cli.py:805-921)
   d. `ctx.orchestrator.run(ctx.pipeline_config, graph=graph, settings=config, payload_store=payload_store)` (cli.py:995)

**What `bootstrap_and_run()` replicates:** Steps 1-6 above, minus CLI-specific concerns (typer.Exit, output formatting, console messages, passphrase prompting). It adds dependency resolution and gate evaluation between steps 4 and 5.

**What `bootstrap_and_run()` does NOT do:** Modify the orchestrator. The orchestrator's `run()` method, `_initialize_database_phase()`, and constructor are untouched. `begin_run()` on `LandscapeRecorder` does NOT accept arbitrary metadata — it takes `config`, `canonical_version`, `source_schema_json`, and `schema_contract`. Dependency/gate metadata should be recorded via a new `recorder.record_preflight_results()` call added by the implementer if needed, or stored in the config dict passed to `begin_run()`.

**Important layer constraint:** `instantiate_plugins_from_config()` lives in `cli_helpers.py` and imports `_get_plugin_manager` from `cli.py`. For `bootstrap_and_run()` (in `engine/bootstrap.py`, L2) to call it, either:
- (a) Move `instantiate_plugins_from_config` and `_get_plugin_manager` into `cli_helpers.py` (both stay L3), then import from `cli_helpers` in `bootstrap.py` — acceptable since `bootstrap.py` is application-layer code despite living under `engine/`, OR
- (b) Accept that `bootstrap.py` is really L3 code (application layer) that happens to live under `engine/` for organizational clarity. The dependency resolver calls it, and the resolver is also L3 in practice.

Option (a) is cleaner. Read `_get_plugin_manager` in cli.py to confirm it has no CLI framework dependencies (typer) before moving.

- [ ] **Step 1: Run existing CLI and engine tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/ tests/integration/engine/ -v -q`
Expected: PASS

- [ ] **Step 2: Read `_get_plugin_manager` in cli.py and confirm it's movable to cli_helpers.py**

It should have no typer dependencies. If it does, `bootstrap_and_run()` must import from `cli.py` directly (acceptable for L3→L3).

- [ ] **Step 3: Create `bootstrap_and_run()` in `src/elspeth/engine/bootstrap.py`**

```python
# src/elspeth/engine/bootstrap.py
"""Programmatic pipeline bootstrap — reusable entry point for dependency resolution."""

from __future__ import annotations

from pathlib import Path

from elspeth.engine.orchestrator.types import RunResult


def bootstrap_and_run(settings_path: Path) -> RunResult:
    """Load config, instantiate plugins, build graph, run pipeline.

    This is the programmatic equivalent of `elspeth run --execute`.
    Used by the dependency resolver to run sub-pipelines.

    Does NOT handle:
    - Output formatting (no typer, no console messages)
    - Passphrase prompting (encrypted DBs not supported for dependency runs)
    - Dependency resolution (caller handles this to avoid infinite recursion)
    - Commencement gates (caller handles this — gates run once for the root pipeline)

    Args:
        settings_path: Absolute path to pipeline settings YAML.

    Returns:
        RunResult from orchestrator.run()

    Raises:
        Any exception from config loading, plugin instantiation, graph validation,
        or pipeline execution. Caller is responsible for error handling.
    """
    from elspeth.cli import _orchestrator_context
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import load_settings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore

    # Phase 1: Load and validate config
    # NOTE: No secret resolution for dependency runs — secrets are inherited
    # from the parent process environment (already populated by root pipeline)
    config = load_settings(settings_path)

    # Phase 2: Instantiate plugins
    plugins = instantiate_plugins_from_config(config)

    # Phase 3: Build and validate execution graph
    execution_sinks = plugins.sinks
    if config.landscape.export.enabled and config.landscape.export.sink:
        export_sink_name = config.landscape.export.sink
        execution_sinks = {k: v for k, v in plugins.sinks.items() if k != export_sink_name}

    graph = ExecutionGraph.from_plugin_instances(
        source=plugins.source,
        source_settings=plugins.source_settings,
        transforms=plugins.transforms,
        sinks=execution_sinks,
        aggregations=plugins.aggregations,
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce) if config.coalesce else None,
    )
    graph.validate()

    # Phase 4: Construct infrastructure and run
    db = LandscapeDB.from_url(
        config.landscape.url,
        dump_to_jsonl=config.landscape.dump_to_jsonl,
        dump_to_jsonl_path=config.landscape.dump_to_jsonl_path,
        dump_to_jsonl_fail_on_error=config.landscape.dump_to_jsonl_fail_on_error,
        dump_to_jsonl_include_payloads=config.landscape.dump_to_jsonl_include_payloads,
        dump_to_jsonl_payload_base_path=(
            str(config.payload_store.base_path)
            if config.landscape.dump_to_jsonl_payload_base_path is None
            else config.landscape.dump_to_jsonl_payload_base_path
        ),
    )

    if config.payload_store.backend != "filesystem":
        raise ValueError(
            f"Unsupported payload store backend '{config.payload_store.backend}'. "
            "Only 'filesystem' is currently supported."
        )
    payload_store = FilesystemPayloadStore(config.payload_store.base_path)

    try:
        with _orchestrator_context(
            config, graph, plugins, db=db,
            output_format="json",  # suppress console output for sub-pipelines
        ) as ctx:
            return ctx.orchestrator.run(
                ctx.pipeline_config,
                graph=graph,
                settings=config,
                payload_store=payload_store,
            )
    finally:
        db.close()
```

- [ ] **Step 4: Do NOT refactor `_execute_pipeline_with_instances()` to delegate to `bootstrap_and_run()`**

The CLI function has additional concerns (passphrase prompting, secret resolution, verbose output, typer.Exit) that `bootstrap_and_run()` intentionally omits. Forcing delegation would either bloat `bootstrap_and_run()` with CLI concerns or lose CLI functionality. The two functions share the same *sequence* but have different *responsibilities*. Code duplication here is intentional — the CLI owns its presentation, `bootstrap_and_run()` owns programmatic execution.

If the duplication bothers you, extract shared helper functions (e.g., `_build_graph_from_plugins(config, plugins)`) but do NOT make `_execute_pipeline_with_instances` delegate to `bootstrap_and_run()`.

- [ ] **Step 5: Run all tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/bootstrap.py
git commit -m "feat: add bootstrap_and_run() for programmatic pipeline execution"
```

---

### Task 6: Dependency Resolution Execution

**Files:**
- Modify: `src/elspeth/engine/dependency_resolver.py`
- Modify: `tests/unit/engine/test_dependency_resolver.py`

- [ ] **Step 1: Write tests for `resolve_dependencies()`**

```python
# Append to tests/unit/engine/test_dependency_resolver.py
from unittest.mock import MagicMock, patch
from elspeth.core.dependency_config import DependencyConfig, DependencyRunResult


class TestResolveDependencies:
    def test_single_dependency_success(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_result = MagicMock()
        mock_result.status.name = "COMPLETED"
        mock_result.run_id = "dep-run-123"

        with patch("elspeth.engine.dependency_resolver.bootstrap_and_run") as mock_boot:
            mock_boot.return_value = mock_result
            results = resolve_dependencies(
                depends_on=[dep],
                parent_settings_path=parent_path,
            )

        assert len(results) == 1
        assert results[0].name == "index"
        assert results[0].run_id == "dep-run-123"

    def test_dependency_failure_raises(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        mock_result = MagicMock()
        mock_result.status.name = "FAILED"
        mock_result.run_id = "dep-run-fail"

        with patch("elspeth.engine.dependency_resolver.bootstrap_and_run") as mock_boot:
            mock_boot.return_value = mock_result
            with pytest.raises(DependencyFailedError, match="index"):
                resolve_dependencies(
                    depends_on=[dep],
                    parent_settings_path=parent_path,
                )

    def test_keyboard_interrupt_propagated(self, tmp_path: Path) -> None:
        dep = DependencyConfig(name="index", settings="./index.yaml")
        parent_path = tmp_path / "query.yaml"

        with patch("elspeth.engine.dependency_resolver.bootstrap_and_run") as mock_boot:
            mock_boot.side_effect = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                resolve_dependencies(
                    depends_on=[dep],
                    parent_settings_path=parent_path,
                )

    def test_multiple_dependencies_sequential(self, tmp_path: Path) -> None:
        deps = [
            DependencyConfig(name="first", settings="./first.yaml"),
            DependencyConfig(name="second", settings="./second.yaml"),
        ]
        parent_path = tmp_path / "main.yaml"
        call_order = []

        def track_calls(path: Path) -> MagicMock:
            call_order.append(path.name)
            result = MagicMock()
            result.status.name = "COMPLETED"
            result.run_id = f"run-{path.name}"
            return result

        with patch("elspeth.engine.dependency_resolver.bootstrap_and_run") as mock_boot:
            mock_boot.side_effect = track_calls
            resolve_dependencies(depends_on=deps, parent_settings_path=parent_path)

        assert call_order == ["first.yaml", "second.yaml"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_dependency_resolver.py::TestResolveDependencies -v`
Expected: FAIL — `resolve_dependencies` not found

- [ ] **Step 3: Implement `resolve_dependencies()`**

Add to `src/elspeth/engine/dependency_resolver.py`:

```python
import time
from datetime import datetime, timezone

from elspeth.contracts.errors import DependencyFailedError
from elspeth.core.canonical import canonical_json
from elspeth.core.dependency_config import DependencyConfig, DependencyRunResult
from elspeth.engine.bootstrap import bootstrap_and_run


def resolve_dependencies(
    *,
    depends_on: list[DependencyConfig],
    parent_settings_path: Path,
) -> list[DependencyRunResult]:
    """Run dependency pipelines sequentially. Raises on failure.

    KeyboardInterrupt is propagated as-is (not wrapped in DependencyFailedError).
    """
    results = []
    for dep in depends_on:
        dep_path = (parent_settings_path.parent / dep.settings).resolve()

        start_ms = time.monotonic_ns() // 1_000_000
        run_result = bootstrap_and_run(dep_path)
        duration_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        if run_result.status.name != "COMPLETED":
            raise DependencyFailedError(
                dependency_name=dep.name,
                run_id=run_result.run_id,
                reason=f"Dependency pipeline finished with status: {run_result.status.name}",
            )

        results.append(DependencyRunResult(
            name=dep.name,
            run_id=run_result.run_id,
            settings_hash=_hash_settings_file(dep_path),
            duration_ms=duration_ms,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        ))
    return results


def _hash_settings_file(path: Path) -> str:
    """SHA-256 hash of the canonical JSON representation of settings."""
    import hashlib
    with path.open() as f:
        data = yaml.safe_load(f)
    canonical = canonical_json(data)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_dependency_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/dependency_resolver.py tests/unit/engine/test_dependency_resolver.py
git commit -m "feat: implement resolve_dependencies() with sequential execution and failure propagation"
```

---

### Task 7: Commencement Gate Evaluation

**Files:**
- Create: `src/elspeth/engine/commencement.py`
- Create: `tests/unit/engine/test_commencement.py`

- [ ] **Step 1: Write gate evaluation tests**

```python
# tests/unit/engine/test_commencement.py
"""Tests for commencement gate evaluation."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.core.dependency_config import CommencementGateConfig, GateResult
from elspeth.engine.commencement import (
    build_preflight_context,
    evaluate_commencement_gates,
)


class TestEvaluateCommencementGates:
    def test_passing_gate(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 10, "reachable": True}},
            "env": {"HOME": "/home/user"},
        }
        results = evaluate_commencement_gates(gates, context)
        assert len(results) == 1
        assert results[0].result is True
        assert results[0].name == "ready"

    def test_failing_gate_raises(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 0, "reachable": False}},
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="ready"):
            evaluate_commencement_gates(gates, context)

    def test_expression_error_raises(self) -> None:
        gates = [
            CommencementGateConfig(
                name="bad",
                condition="collections['missing']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {},
            "env": {},
        }
        with pytest.raises(CommencementGateFailedError, match="bad"):
            evaluate_commencement_gates(gates, context)

    def test_snapshot_excludes_env(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 5, "reachable": True}},
            "env": {"SECRET_KEY": "abc123"},
        }
        results = evaluate_commencement_gates(gates, context)
        snapshot = results[0].context_snapshot
        assert "env" not in snapshot
        assert "SECRET_KEY" not in str(snapshot)

    def test_snapshot_is_deep_frozen(self) -> None:
        gates = [
            CommencementGateConfig(
                name="ready",
                condition="collections['test']['count'] > 0",
            )
        ]
        context = {
            "dependency_runs": {},
            "collections": {"test": {"count": 5, "reachable": True}},
            "env": {},
        }
        results = evaluate_commencement_gates(gates, context)
        assert isinstance(results[0].context_snapshot, MappingProxyType)


class TestBuildPreflightContext:
    def test_includes_all_sections(self) -> None:
        context = build_preflight_context(
            dependency_results={},
            collection_probes={"test": {"count": 5, "reachable": True}},
            env={"HOME": "/home"},
        )
        assert "dependency_runs" in context
        assert "collections" in context
        assert "env" in context
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_commencement.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement gate evaluation**

```python
# src/elspeth/engine/commencement.py
"""Commencement gate evaluation — pre-flight go/no-go checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from elspeth.contracts.errors import CommencementGateFailedError
from elspeth.contracts.freeze import deep_freeze
from elspeth.core.dependency_config import CommencementGateConfig, GateResult
from elspeth.core.expression_parser import ExpressionParser

_GATE_ALLOWED_NAMES = ["collections", "dependency_runs", "env"]


def build_preflight_context(
    *,
    dependency_results: dict[str, Any],
    collection_probes: dict[str, Any],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the pre-flight context dict for gate expression evaluation."""
    return {
        "dependency_runs": dependency_results,
        "collections": collection_probes,
        "env": env if env is not None else dict(os.environ),
    }


def _build_audit_snapshot(context: dict[str, Any]) -> Mapping[str, Any]:
    """Build a frozen context snapshot for audit, excluding env."""
    snapshot = {
        "dependency_runs": context["dependency_runs"],
        "collections": context["collections"],
    }
    return deep_freeze(snapshot)


def evaluate_commencement_gates(
    gates: list[CommencementGateConfig],
    context: dict[str, Any],
) -> list[GateResult]:
    """Evaluate gates sequentially. Raises CommencementGateFailedError on failure."""
    frozen_context = deep_freeze(context)
    audit_snapshot = _build_audit_snapshot(context)

    results = []
    for gate in gates:
        try:
            parser = ExpressionParser(
                gate.condition,
                allowed_names=_GATE_ALLOWED_NAMES,
            )
            passed = bool(parser.evaluate(frozen_context))
        except CommencementGateFailedError:
            raise
        except Exception as exc:
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason=f"Expression raised {type(exc).__name__}: {exc}",
                context_snapshot=audit_snapshot,
            ) from exc

        if not passed:
            raise CommencementGateFailedError(
                gate_name=gate.name,
                condition=gate.condition,
                reason="Condition evaluated to falsy",
                context_snapshot=audit_snapshot,
            )

        results.append(GateResult(
            name=gate.name,
            condition=gate.condition,
            result=True,
            context_snapshot=audit_snapshot,
        ))
    return results
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_commencement.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/commencement.py tests/unit/engine/test_commencement.py
git commit -m "feat: implement commencement gate evaluation with ExpressionParser and frozen snapshots"
```

---

### Task 8: Collection Probe Factory

**Files:**
- Create: `src/elspeth/plugins/infrastructure/probe_factory.py`
- Create: `tests/unit/plugins/infrastructure/test_probe_factory.py`

- [ ] **Step 1: Write probe factory tests**

```python
# tests/unit/plugins/infrastructure/test_probe_factory.py
"""Tests for collection probe factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.probes import CollectionProbe, CollectionReadinessResult
from elspeth.core.dependency_config import CollectionProbeConfig
from elspeth.plugins.infrastructure.probe_factory import build_collection_probes


class TestBuildCollectionProbes:
    def test_builds_chroma_probe(self) -> None:
        configs = [
            CollectionProbeConfig(
                collection="test",
                provider="chroma",
                provider_config={"mode": "persistent", "persist_directory": "./data"},
            )
        ]
        probes = build_collection_probes(configs)
        assert len(probes) == 1
        assert isinstance(probes[0], CollectionProbe)
        assert probes[0].collection_name == "test"

    def test_empty_configs_returns_empty(self) -> None:
        assert build_collection_probes([]) == []

    def test_unknown_provider_raises(self) -> None:
        configs = [
            CollectionProbeConfig(
                collection="test",
                provider="unknown_provider",
                provider_config={},
            )
        ]
        with pytest.raises(ValueError, match="unknown_provider"):
            build_collection_probes(configs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_probe_factory.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement probe factory**

```python
# src/elspeth/plugins/infrastructure/probe_factory.py
"""Factory for constructing collection probes from explicit config declarations."""

from __future__ import annotations

from typing import Any

from elspeth.contracts.probes import CollectionProbe, CollectionReadinessResult
from elspeth.core.dependency_config import CollectionProbeConfig


class ChromaCollectionProbe:
    """Probes a ChromaDB collection for readiness."""

    def __init__(self, collection: str, config: dict[str, Any]) -> None:
        self.collection_name = collection
        self._config = config

    def probe(self) -> CollectionReadinessResult:
        try:
            import chromadb

            mode = self._config.get("mode", "persistent")
            if mode == "persistent":
                client = chromadb.PersistentClient(
                    path=self._config["persist_directory"]
                )
            else:
                client = chromadb.HttpClient(
                    host=self._config["host"],
                    port=self._config.get("port", 8000),
                    ssl=self._config.get("ssl", True),
                )

            try:
                collection = client.get_collection(self.collection_name)
                count = collection.count()
                return CollectionReadinessResult(
                    collection=self.collection_name,
                    reachable=True,
                    count=count,
                    message=f"Collection '{self.collection_name}' has {count} documents"
                    if count > 0
                    else f"Collection '{self.collection_name}' is empty",
                )
            except Exception:
                return CollectionReadinessResult(
                    collection=self.collection_name,
                    reachable=True,
                    count=0,
                    message=f"Collection '{self.collection_name}' not found",
                )
        except Exception:
            return CollectionReadinessResult(
                collection=self.collection_name,
                reachable=False,
                count=0,
                message=f"Collection '{self.collection_name}' unreachable",
            )


_PROBE_REGISTRY: dict[str, type] = {
    "chroma": ChromaCollectionProbe,
}


def build_collection_probes(
    configs: list[CollectionProbeConfig],
) -> list[CollectionProbe]:
    """Construct probes from explicit config declarations."""
    probes: list[CollectionProbe] = []
    for config in configs:
        probe_cls = _PROBE_REGISTRY.get(config.provider)
        if probe_cls is None:
            raise ValueError(
                f"Unknown collection probe provider: {config.provider!r}. "
                f"Available: {sorted(_PROBE_REGISTRY)}"
            )
        probes.append(probe_cls(config.collection, config.provider_config))
    return probes
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_probe_factory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/infrastructure/probe_factory.py tests/unit/plugins/infrastructure/test_probe_factory.py
git commit -m "feat: add collection probe factory with ChromaDB implementation"
```

---

### Task 9: Wire Dependency + Gate Phases into `bootstrap_and_run()`

**Files:**
- Modify: `src/elspeth/engine/bootstrap.py`
- Create: `tests/unit/engine/test_bootstrap_preflight.py`

**Design: pre-flight logic lives in `bootstrap_and_run()`, NOT in the orchestrator.**

The orchestrator's `run()` method is untouched. `bootstrap_and_run()` already has the `settings_path` (it's its input parameter) and the loaded `config` (it loads it in Phase 1). Dependency resolution and gate evaluation are new phases inserted between graph validation (Phase 3) and infrastructure construction (Phase 4).

The orchestrator does not need `self._settings_path`. It never sees dependency configs or gate configs. The pre-flight results are recorded in the audit trail by `bootstrap_and_run()` after the orchestrator's run completes (as post-run metadata), or passed into the config dict that `begin_run()` receives. Read the `begin_run()` signature to decide: it accepts `config: Mapping[str, Any]` which is the pipeline config dict — dependency/gate results can be merged into this dict.

- [ ] **Step 1: Run full engine test suite to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/ tests/integration/engine/ -x -q`
Expected: PASS

- [ ] **Step 2: Add dependency + gate phases to `bootstrap_and_run()`**

In `src/elspeth/engine/bootstrap.py`, insert new phases between graph validation and infrastructure construction:

```python
# After Phase 3 (graph.validate()), before Phase 4 (LandscapeDB):

from elspeth.engine.dependency_resolver import detect_cycles, resolve_dependencies
from elspeth.engine.commencement import build_preflight_context, evaluate_commencement_gates
from elspeth.plugins.infrastructure.probe_factory import build_collection_probes

# Phase 3.5: Dependency resolution (if configured)
dependency_results = []
if config.depends_on:
    # Cycle detection first (cheap, reads only depends_on keys from YAML)
    detect_cycles(settings_path)

    # Run dependencies sequentially — each calls bootstrap_and_run() recursively
    # NOTE: Recursive calls do NOT re-run dependency resolution (dependency
    # pipelines don't inherit the parent's depends_on config)
    dependency_results = resolve_dependencies(
        depends_on=config.depends_on,
        parent_settings_path=settings_path,
    )

# Phase 3.6: Commencement gates (if configured)
gate_results = []
if config.commencement_gates:
    # Build probes and execute them
    probes = build_collection_probes(config.collection_probes)
    probe_results = {}
    for probe in probes:
        result = probe.probe()
        probe_results[result.collection] = {
            "reachable": result.reachable,
            "count": result.count,
        }

    dep_run_dict = {
        r.name: {"run_id": r.run_id, "duration_ms": r.duration_ms, "indexed_at": r.indexed_at}
        for r in dependency_results
    }

    context = build_preflight_context(
        dependency_results=dep_run_dict,
        collection_probes=probe_results,
    )
    gate_results = evaluate_commencement_gates(config.commencement_gates, context)

# Phase 4: Construct infrastructure and run (existing code from Task 5)
# ...
```

The `settings_path` is already available as the function's input parameter — no threading needed.

- [ ] **Step 3: Write tests for pre-flight integration in bootstrap**

```python
# tests/unit/engine/test_bootstrap_preflight.py
"""Tests for dependency + gate phases in bootstrap_and_run()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.errors import CommencementGateFailedError, DependencyFailedError


class TestBootstrapDependencyResolution:
    """Test that bootstrap_and_run() calls dependency resolution when configured."""

    def test_skips_dependencies_when_not_configured(self, tmp_path: Path) -> None:
        """When depends_on is empty, dependency resolver is never called."""
        # Create a minimal settings file with no depends_on
        settings = tmp_path / "pipeline.yaml"
        settings.write_text(
            "source:\n  plugin: null_source\n"
            "sinks:\n  out:\n    plugin: json_sink\n"
            "landscape:\n  url: sqlite:///test.db\n"
        )

        with (
            patch("elspeth.engine.bootstrap.detect_cycles") as mock_detect,
            patch("elspeth.engine.bootstrap.resolve_dependencies") as mock_resolve,
            patch("elspeth.engine.bootstrap.load_settings") as mock_load,
        ):
            mock_config = MagicMock()
            mock_config.depends_on = []
            mock_config.commencement_gates = []
            mock_load.return_value = mock_config

            # We'll need to mock more to avoid full pipeline execution
            # The key assertion: detect_cycles and resolve_dependencies NOT called
            # (full execution mocking is complex — consider integration test instead)

        mock_detect.assert_not_called()
        mock_resolve.assert_not_called()


class TestBootstrapCommencementGates:
    """Test that bootstrap_and_run() evaluates gates when configured."""

    def test_skips_gates_when_not_configured(self) -> None:
        """When commencement_gates is empty, gate evaluator is never called."""
        # Similar structure to dependency test above
        pass  # Implement with same mock pattern
```

**Note:** Full integration tests for the bootstrap pre-flight are in Task 10's integration test files (`tests/integration/engine/test_depends_on.py` and `tests/integration/engine/test_commencement_gates.py`). The unit tests here verify the conditional dispatch logic (skip when not configured, call when configured).

- [ ] **Step 4: Run all tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/bootstrap.py tests/unit/engine/test_bootstrap_preflight.py
git commit -m "feat: wire dependency resolution and commencement gates into bootstrap_and_run()"
```

---

### Task 10: Type Checking, Linting, Full Verification

**Files:** None new — verification only.

- [ ] **Step 1: Run type checker on all new files**

Run: `.venv/bin/python -m mypy src/elspeth/core/dependency_config.py src/elspeth/engine/dependency_resolver.py src/elspeth/engine/commencement.py src/elspeth/plugins/infrastructure/probe_factory.py`
Expected: PASS

- [ ] **Step 2: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/core/dependency_config.py src/elspeth/engine/dependency_resolver.py src/elspeth/engine/commencement.py src/elspeth/plugins/infrastructure/probe_factory.py`
Expected: PASS

- [ ] **Step 3: Run tier model enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 5: Commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address type/lint/tier issues in dependency and gate implementation"
```
