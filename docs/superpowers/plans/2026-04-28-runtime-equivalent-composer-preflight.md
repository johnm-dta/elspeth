# Runtime-Equivalent Composer Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authoritative composer preflight path so composer preview, final assistant responses, persisted composer state, YAML export, `/validate`, and `/execute` agree about runtime-blocking pipeline failures before any side-effectful run starts.

**Architecture:** Keep `CompositionState.validate()` as the pure, fast authoring validator. Add a shared web execution preflight boundary that generates composer YAML, resolves runtime-relative paths, resolves deferred secret references in memory, loads settings with `load_settings_from_yaml_string()`, instantiates plugins, builds an `ExecutionGraph`, then runs `graph.validate()` and `graph.validate_edge_compatibility()` through the same production path as execution. Composer code consumes that boundary as an additional authoritative result; it must not copy engine rules into `CompositionState.validate()`.

**Tech Stack:** Python 3.12+, FastAPI route handlers, Pydantic response schemas, SQLAlchemy Core session tables, existing ELSPETH composer state/YAML generator, `load_settings_from_yaml_string()`, `instantiate_plugins_from_config()`, `ExecutionGraph.from_plugin_instances()`, pytest.

---

## Issue

Tracker issue: `elspeth-34baf10c01` - Runtime-equivalent composer preflight / dry-run validation before complete-and-valid claims.

The defect is not that `CompositionState.validate()` needs more local rules. It is intentionally pure and context-free. The defect is that composer prompt/tool/final/persistence surfaces treat that authoring validator as authoritative even though `/validate` and `/execute` use stronger runtime setup paths.

## Reviewer Fixes Folded In

- All composer-to-DB state write paths must use one persisted-validation helper: `send_message`, `recompose`, `_handle_convergence_error`, and `_handle_plugin_crash`.
- Do not use a natural-language completion regex. Final preflight is deterministic: run when the state changed in the current compose turn, or reuse a runtime preflight already returned by `preview_pipeline`.
- If the visible assistant message is replaced with a synthetic preflight-failure message, persist the raw model text separately in `chat_messages.raw_content`; do not expose it through `ChatMessageResponse`.
- Unexpected runtime-preflight exceptions must not be laundered into user-fixable `ValidationResult` objects. Expected runtime validation blockers remain `ValidationResult(is_valid=False, ...)`; unexpected programmer/infrastructure failures become a typed composer error with partial-state preservation.
- YAML export must validate the exact state snapshot that it serializes, not re-read current state through `execution_service.validate(session_id)`.
- Scenario tests must use the real helpers in `tests/integration/pipeline/test_composer_llm_eval_characterization.py`: `_scenario_2_files(tmp_path)` and `_aggregation_state(source_path, output_path, *, ...)`.
- Normal composer validation/preflight pass/fail/exception rates are telemetry, not `slog`; use OpenTelemetry counters with bounded attributes and keep session-level correlation in persisted session state.
- The preflight validator and runtime graph setup share `RUNTIME_GRAPH_VALIDATION_CHECKS`, with a test that successful `validate_pipeline()` surfaces every declared runtime graph check.
- `ComposerRuntimePreflightError` must be added to the closed catch-order CI enforcer registry when the exception class is introduced.
- `ComposerRuntimePreflightError._FROZEN_ATTRS` must include `original_exc`, matching sibling composer error classes so captured exception identity cannot be rewritten after construction.
- `ComposerRuntimePreflightError.original_exc` is typed as `Exception`, not `BaseException`, because capture sites handle ordinary `Exception` subclasses such as `TimeoutError` and this matches sibling composer error patterns.
- Persisted-state helpers must accept a single runtime-preflight outcome parameter, not independent `runtime_preflight` and `preflight_failed` knobs. Use a typed sentinel for internal preflight failure so invalid combinations are unrepresentable.
- Existing execution-service tests have concrete patch-target pairs to migrate from `elspeth.web.execution.service.*` to `elspeth.web.execution.preflight.*`; enumerate them and add a persistent pytest guard so future stale patch sites are not missed.
- Add tests for all `_runtime_preflight_failure_message()` branches, the authoring-only `preview_pipeline` default, and reuse of preview runtime preflight when the state is unchanged at finalization.
- Forked sessions must preserve `chat_messages.raw_content` for copied historical messages; this keeps raw assistant provenance attached to the conversation history after fork.
- Adding `chat_messages.raw_content` is a schema-bootstrap change. Existing dev/staging `sessions.db` files must be reset through `docs/guides/session-db-reset.md`; before staging rollout, re-confirm Landscape has no `session_id`, `chat_message_id`, or `composition_state_id` references so deleting the session DB cannot orphan audit rows.
- Low-priority hardening: avoid kwargs-fragile loader spies, assert final-gate structural contracts rather than message copy substrings, test Tier 1 `state.to_dict()` propagation, document raw-content retention, and track deferred frontend-copy/runtime-dispatch follow-ups.
- Runtime graph check names must use named constants plus an explicit order assertion; never unpack `RUNTIME_GRAPH_VALIDATION_CHECKS` positionally into `_CHECK_*` names.
- Audit-integrity coverage must prove exported composer YAML does not contain resolved secret values.
- Patch-target drift checks must be persistent pytest coverage under `tests/unit/scripts/cicd/`, not a one-shot `rg` instruction.
- Runtime preflight must be side-effect-free by construction. Composer preflight must instantiate plugins in `preflight_mode=True`, must not run plugin lifecycle hooks or open network sockets, and must have persistent fake-socket coverage for representative Azure/Dataverse/LLM/RAG/external-sink constructors. Add a separate production-path test proving plugin constructors observe `plugin_preflight_mode_enabled() is True` when `validate_pipeline()` is dispatched through `run_sync_in_worker`; do not rely on parent-task `ContextVar` propagation or the direct-kwarg shortcut as evidence for the threaded path.
- Runtime preflight must be bounded and deduplicated. A compose call gets a `(session_scope, state.version, settings_hash)` scoped cache and an explicit `composer_runtime_preflight_timeout_seconds`; timeout failures are cached for the compose call so repeated preview/final checks do not start more hung worker threads.
- Runtime preflight in-flight deduplication must be transport-neutral. HTTP composer preview and composer MCP `preview_pipeline` must use the same process-local single-flight coordinator when they share a logical session, keyed by `(session_scope, state.version, settings_hash)`, so concurrent preview requests do not race plugin instantiation. Standalone MCP processes cannot share process-local locks with the web process; side-effect-free `preflight_mode=True` is the cross-process safety property.
- Multi-turn composer history must remain self-consistent after synthetic assistant substitution. `composer/prompts.py::build_messages()` feeds the LLM from the route-supplied `chat_history[*]["content"]`, and session routes currently build that from `ChatMessageRecord.content`; therefore intercepted assistant turns need an explicit annotation in LLM history while `raw_content` remains persisted-only and is never echoed back to the model.

## Scope Check

This plan touches one coherent subsystem: web composer validation and execution preflight. It crosses several files, but all changes serve one testable behavior: generated composer state must be checked by the same pre-execution truth the runtime uses before the system reports or persists it as valid.

Out of scope:

- Running sample data rows.
- Running plugin lifecycle hooks (`on_start`, `load`, `process`, `write`, `close`) or plugin-specific side-effectful dry-run hooks.
- Exposing hidden model reasoning.
- Frontend UI redesign. A follow-up can rename the visible "Errors" heading if runtime-flavored messages need clearer copy.
- Expanding path rewriting beyond existing source/sink option keys. Transform/aggregation path options are a separate pre-existing gap and should be tracked separately if found.

## File Structure

- Create `src/elspeth/web/execution/preflight.py`
  - Own shared, side-effect-free preflight helpers.
  - Move YAML path resolution here from `execution/service.py`.
  - Provide typed plugin/graph setup helpers without collapsing `graph_structure` and `schema_compatibility` validation checks.
  - Provide a non-secret runtime-preflight settings hash used only for per-compose cache keys.

- Create `src/elspeth/web/execution/runtime_preflight.py`
  - Own transport-neutral runtime-preflight single-flight coordination.
  - Deduplicate in-flight validation by logical session, state version, and settings hash across HTTP composer and embedded composer MCP callers in the same process.
  - Return typed success/failure entries so each caller can apply its own HTTP/MCP error semantics.

- Create `src/elspeth/plugins/infrastructure/preflight.py`
  - Own the process-local plugin-instantiation preflight mode context.

- Modify `src/elspeth/cli_helpers.py`
  - Add `instantiate_plugins_from_config(..., preflight_mode: bool = False)`.
  - Keep runtime execution on the default `False`; composer/web validation preflight uses `True`.

- Modify `src/elspeth/web/config.py`
  - Add `composer_runtime_preflight_timeout_seconds` with a positive default.

- Modify `src/elspeth/web/execution/protocol.py`
  - Add a small `ValidationSettings` protocol with `data_dir`.

- Modify `src/elspeth/web/execution/validation.py`
  - Keep `validate_pipeline()` as the public direct-state validation API.
  - Use shared path resolution and runtime graph helpers.
  - Preserve separate check names: `plugin_instantiation`, `graph_structure`, and `schema_compatibility`.

- Modify `src/elspeth/web/execution/service.py`
  - Use the shared path resolver and runtime graph helper in `_execute_locked()` / `_run_pipeline()`.
  - Keep run creation, blob ownership, audit, progress, and orchestration here.

- Modify `src/elspeth/web/composer/tools.py`
  - Add optional runtime preflight callback to `execute_tool()`.
  - Add `ToolResult.runtime_preflight`.
  - Make `preview_pipeline` include both authoring validation and runtime preflight when available.

- Modify `src/elspeth/web/composer/service.py`
  - Store runtime-validation settings.
  - Pass runtime preflight into tools.
  - Track the latest preview preflight result.
  - Add a per-compose runtime-preflight cache keyed by `(session_scope, state.version, settings_hash)`.
  - Use the shared runtime-preflight coordinator to deduplicate concurrent in-flight preflights for the same logical session/state/settings.
  - Apply the explicit preflight timeout outside the general side-effectful tool-call path.
  - Gate final responses deterministically with no text regex.

- Modify `src/elspeth/web/app.py`
  - Create one `RuntimePreflightCoordinator` per web process and inject it into `ComposerServiceImpl`.

- Modify `src/elspeth/composer_mcp/server.py`
  - Thread optional runtime preflight through MCP `preview_pipeline`.
  - Use the same coordinator abstraction and logical session-scope key as HTTP when embedded in-process; use scratch-session scope for standalone MCP.

- Modify `src/elspeth/web/composer/protocol.py`
  - Extend `ComposerResult` with `runtime_preflight` and `raw_assistant_content`.
  - Add `ComposerRuntimePreflightError` for unexpected internal preflight failures.

- Modify `scripts/cicd/enforce_composer_catch_order.py`
  - Add `ComposerRuntimePreflightError` to the closed `ComposerServiceError` subclass map.

- Modify `src/elspeth/web/sessions/models.py`, `protocol.py`, `service.py`, `schemas.py`, and `routes.py`
  - Add nullable `chat_messages.raw_content` for raw model text when visible content is replaced.
  - Persist runtime validation truth through all composer state write paths.
  - Keep `raw_content` out of `ChatMessageResponse`.
  - Keep `raw_content` out of composer LLM history while annotating intercepted assistant turns.
  - Gate YAML export by validating the exact state snapshot being exported.
  - Preserve `raw_content` when `fork_session()` copies historical messages into a fork.

- Tests:
  - `tests/unit/web/execution/test_validation.py`
  - `tests/unit/web/execution/test_preflight_side_effects.py`
  - `tests/unit/web/execution/test_runtime_preflight_coordinator.py`
  - `tests/unit/web/execution/test_service.py`
  - `tests/unit/web/composer/test_tools.py`
  - `tests/unit/web/composer/test_service.py`
  - `tests/unit/composer_mcp/test_server.py`
  - `tests/unit/web/sessions/test_routes.py`
  - `tests/unit/web/sessions/test_fork.py`
  - `tests/unit/scripts/cicd/test_enforce_composer_catch_order.py`
  - `tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py`
  - `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

## Design Rules

- `CompositionState.validate()` remains pure: no settings, no session, no DB, no filesystem, no plugin instantiation.
- Runtime preflight is the authority for "runnable" and "complete".
- `composition_states.is_valid` must have one meaning after this change: runtime-preflight-valid when runtime preflight is available; otherwise false with an explicit validation error if preflight could not be completed.
- Composer preview shows authoring validation plus runtime preflight; it does not hide authoring warnings.
- Composer runtime preflight is not a dry run of external systems. It may parse config, resolve in-memory secret refs, instantiate plugin objects in preflight mode, build the graph, and run graph validators. It must not call plugin lifecycle methods or constructors/factories that open sockets, authenticate against cloud services, initialize process-global provider instrumentation, or create external clients that perform health checks.
- `preflight_mode=True` must preserve validation fidelity: it may defer network/client setup, but it must not skip Pydantic config parsing, schema construction, declared field setup, graph construction, graph validation, or edge compatibility validation.
- Timeout is an await bound, not a thread kill. Python cannot safely cancel a running synchronous constructor; therefore the compose-scoped cache must cache timeout/internal-failure outcomes as well as successful `ValidationResult` outcomes so repeated preview/final checks do not amplify hung workers.
- In-flight runtime-preflight dedupe is process-local and transport-neutral. Do not key it by `"http"` vs `"mcp"`; key it by logical session scope plus state version and settings hash. Do not add a cross-process distributed lock in this PR: standalone MCP and web can still duplicate work when they run in different processes, and that is acceptable only because Task 2 makes preflight plugin construction side-effect-free.
- Expected plugin/config/user pipeline errors become structured validation entries.
- Unexpected internal failures stay internal failures. They must preserve partial composer state where possible, return sanitized 500s, and avoid exposing raw exception text to the client or LLM.
- Secret values must never be logged or persisted in generated YAML. Resolved secret values are in-memory only.
- Tests must use production code paths: `load_settings_from_yaml_string()`, `instantiate_plugins_from_config()`, `ExecutionGraph.from_plugin_instances()`, `graph.validate()`, and `graph.validate_edge_compatibility()`.

---

### Task 1: Shared Preflight Foundations And Path Parity

**Files:**
- Create: `src/elspeth/web/execution/preflight.py`
- Modify: `src/elspeth/web/execution/protocol.py`
- Modify: `src/elspeth/web/execution/validation.py`
- Modify: `src/elspeth/web/execution/service.py`
- Test: `tests/unit/web/execution/test_validation.py`

- [ ] **Step 1: Add failing path-resolution tests**

Append to `tests/unit/web/execution/test_validation.py`:

```python
class TestValidatePipelineRuntimePathResolution:
    @staticmethod
    def _loaded_yaml_from_settings_loader(mock_load: MagicMock) -> str:
        call = mock_load.call_args
        if call.args:
            return call.args[0]
        return call.kwargs["yaml_content"]

    def test_validate_pipeline_resolves_relative_source_and_sink_paths_before_settings_load(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "blobs/session/input.csv"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "outputs/out.csv"},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: main
  options:
    path: blobs/session/input.csv
    on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: outputs/out.csv
"""

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("stop after settings-load input capture")
            validate_pipeline(state, settings, mock_yaml_gen)

        loaded_yaml = self._loaded_yaml_from_settings_loader(mock_load)
        parsed = yaml.safe_load(loaded_yaml)
        assert parsed["source"]["options"]["path"] == "/tmp/test_data/blobs/session/input.csv"
        assert parsed["sinks"]["main"]["options"]["path"] == "/tmp/test_data/outputs/out.csv"

    def test_validate_pipeline_preserves_absolute_paths_before_settings_load(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/tmp/test_data/blobs/input.csv"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": "/tmp/test_data/outputs/out.csv"},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: main
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  main:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""

        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as mock_load:
            mock_load.side_effect = ValueError("stop after settings-load input capture")
            validate_pipeline(state, settings, mock_yaml_gen)

        loaded_yaml = self._loaded_yaml_from_settings_loader(mock_load)
        parsed = yaml.safe_load(loaded_yaml)
        assert parsed["source"]["options"]["path"] == "/tmp/test_data/blobs/input.csv"
        assert parsed["sinks"]["main"]["options"]["path"] == "/tmp/test_data/outputs/out.csv"
```

- [ ] **Step 2: Run the failing path tests**

Run:

```bash
uv run pytest -q tests/unit/web/execution/test_validation.py::TestValidatePipelineRuntimePathResolution
```

Expected: the relative-path test fails because `validate_pipeline()` currently passes relative paths to `load_settings_from_yaml_string()`.

- [ ] **Step 3: Add a validation settings protocol**

Modify `src/elspeth/web/execution/protocol.py`:

```python
from typing import Any, Protocol, runtime_checkable


class ValidationSettings(Protocol):
    """Settings needed by direct runtime preflight validation."""

    @property
    def data_dir(self) -> Any: ...
```

Keep the existing `YamlGenerator` and `ExecutionService` protocols.

- [ ] **Step 4: Create `preflight.py` with path resolution and typed graph helpers**

Create `src/elspeth/web/execution/preflight.py`:

```python
"""Shared runtime preflight helpers for web validation and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elspeth.cli_helpers import PluginBundle, instantiate_plugins_from_config
from elspeth.core.dag.graph import ExecutionGraph


RUNTIME_CHECK_PLUGIN_INSTANTIATION = "plugin_instantiation"
RUNTIME_CHECK_GRAPH_STRUCTURE = "graph_structure"
RUNTIME_CHECK_SCHEMA_COMPATIBILITY = "schema_compatibility"

RUNTIME_GRAPH_VALIDATION_CHECKS: tuple[str, str, str] = (
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
)
assert RUNTIME_GRAPH_VALIDATION_CHECKS == (
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
)


@dataclass(slots=True)
class RuntimeGraphBundle:
    """Transient runtime setup result.

    Not frozen: ExecutionGraph is mutable runtime state. This object is not
    persisted and should not cross request boundaries.
    """

    plugin_bundle: PluginBundle
    graph: ExecutionGraph


def resolve_runtime_yaml_paths(pipeline_yaml: str, data_dir: str) -> str:
    """Rewrite relative source/sink paths in pipeline YAML to absolute paths."""
    from elspeth.web.paths import resolve_data_path

    if not isinstance(pipeline_yaml, str):
        raise TypeError(f"YamlGenerator.generate_yaml() must return str; got {type(pipeline_yaml).__name__}")

    config = yaml.safe_load(pipeline_yaml)
    if not isinstance(config, dict):
        raise TypeError(f"YAML generator produced non-dict top-level value (got {type(config).__name__})")

    source = config.get("source")
    if source is not None:
        if not isinstance(source, dict):
            raise TypeError(f"YAML generator produced non-dict 'source' value (got {type(source).__name__})")
        opts = source.get("options")
        if opts is not None:
            if not isinstance(opts, dict):
                raise TypeError(f"YAML generator produced non-dict 'source.options' value (got {type(opts).__name__})")
            for key in ("path", "file"):
                if key in opts and not Path(str(opts[key])).is_absolute():
                    opts[key] = str(resolve_data_path(str(opts[key]), data_dir))

    sinks = config.get("sinks")
    if sinks is not None:
        if not isinstance(sinks, dict):
            raise TypeError(f"YAML generator produced non-dict 'sinks' value (got {type(sinks).__name__})")
        for sink_name, sink_cfg in sinks.items():
            if sink_cfg is not None:
                if not isinstance(sink_cfg, dict):
                    raise TypeError(f"YAML generator produced non-dict sink '{sink_name}' value (got {type(sink_cfg).__name__})")
                opts = sink_cfg.get("options")
                if opts is not None:
                    if not isinstance(opts, dict):
                        raise TypeError(f"YAML generator produced non-dict 'sinks.{sink_name}.options' value (got {type(opts).__name__})")
                    for key in ("path", "file"):
                        if key in opts and not Path(str(opts[key])).is_absolute():
                            opts[key] = str(resolve_data_path(str(opts[key]), data_dir))

    return yaml.dump(config, default_flow_style=False)


def runtime_preflight_settings_hash(settings: Any) -> str:
    """Return a non-secret hash of settings that affect runtime preflight.

    Current ValidationSettings exposes only data_dir. If new settings affect
    validation later, add them here deliberately and keep secret-bearing fields
    out of the payload.
    """
    payload = {
        "data_dir": str(Path(settings.data_dir).expanduser().resolve()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def instantiate_runtime_plugins(settings: Any, *, preflight_mode: bool = False) -> PluginBundle:
    """Instantiate configured plugins through the production helper."""
    return instantiate_plugins_from_config(settings, preflight_mode=preflight_mode)


def build_runtime_graph(settings: Any, bundle: PluginBundle) -> ExecutionGraph:
    """Build an ExecutionGraph through the production graph factory."""
    return ExecutionGraph.from_plugin_instances(
        source=bundle.source,
        source_settings=bundle.source_settings,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
    )


def build_validated_runtime_graph(settings: Any) -> RuntimeGraphBundle:
    """Instantiate runtime plugins, build the graph, and run both runtime graph checks.

    Used by execution before running the pipeline, so this must use normal
    runtime mode. Composer/web validation calls instantiate_runtime_plugins()
    directly with preflight_mode=True instead.
    """
    bundle = instantiate_runtime_plugins(settings, preflight_mode=False)
    graph = build_runtime_graph(settings, bundle)
    graph.validate()
    graph.validate_edge_compatibility()
    return RuntimeGraphBundle(plugin_bundle=bundle, graph=graph)
```

- [ ] **Step 5: Use path resolution in validation and execution**

In `src/elspeth/web/execution/validation.py`, import:

```python
from elspeth.web.execution.preflight import resolve_runtime_yaml_paths
from elspeth.web.execution.protocol import ValidationSettings, YamlGenerator
```

Change the `settings` parameter type from `WebSettings` to `ValidationSettings`.

After YAML generation in `validate_pipeline()`:

```python
    pipeline_yaml = yaml_generator.generate_yaml(state)
    pipeline_yaml = resolve_runtime_yaml_paths(pipeline_yaml, str(settings.data_dir))
```

In `src/elspeth/web/execution/service.py`, import:

```python
from elspeth.web.execution.preflight import resolve_runtime_yaml_paths
```

Replace the `_execute_locked()` call to the local path resolver with:

```python
        pipeline_yaml = resolve_runtime_yaml_paths(pipeline_yaml, str(self._settings.data_dir))
```

Delete the old `_resolve_yaml_paths()` function from `execution/service.py` after replacing all callers.

- [ ] **Step 6: Run path tests**

Run:

```bash
uv run pytest -q tests/unit/web/execution/test_validation.py::TestValidatePipelineRuntimePathResolution
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/execution/preflight.py src/elspeth/web/execution/protocol.py src/elspeth/web/execution/validation.py src/elspeth/web/execution/service.py tests/unit/web/execution/test_validation.py
git commit -m "fix(web): resolve composer yaml paths during validation preflight"
```

---

### Task 2: Share Runtime Plugin And Graph Setup Without Collapsing Checks Or Opening Sockets

**Files:**
- Create: `src/elspeth/plugins/infrastructure/preflight.py`
- Modify: `src/elspeth/cli_helpers.py`
- Modify: `src/elspeth/web/execution/validation.py`
- Modify: `src/elspeth/web/execution/service.py`
- Test: `tests/unit/web/execution/test_preflight_side_effects.py`
- Test: `tests/unit/web/execution/test_validation.py`
- Test: `tests/unit/web/execution/test_service.py`
- Test: `tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py`

- [ ] **Step 1: Add validation tests for separate graph and schema checks**

Append to `tests/unit/web/execution/test_validation.py`:

```python
class TestValidatePipelineRuntimeCheckBoundaries:
    def test_runtime_graph_validation_check_order_matches_named_constants(self) -> None:
        from elspeth.web.execution.preflight import (
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
            RUNTIME_GRAPH_VALIDATION_CHECKS,
        )

        assert RUNTIME_GRAPH_VALIDATION_CHECKS == (
            RUNTIME_CHECK_PLUGIN_INSTANTIATION,
            RUNTIME_CHECK_GRAPH_STRUCTURE,
            RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
        )

    def test_validate_pipeline_success_surfaces_declared_runtime_graph_checks(self) -> None:
        from elspeth.web.execution.preflight import RUNTIME_GRAPH_VALIDATION_CHECKS

        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_graph = MagicMock()

        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", return_value=MagicMock()),
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins", return_value=MagicMock()) as mock_instantiate,
            patch("elspeth.web.execution.validation.build_runtime_graph", return_value=fake_graph),
        ):
            result = validate_pipeline(state, settings, mock_yaml_gen)

        passed_names = {check.name for check in result.checks if check.passed}
        assert set(RUNTIME_GRAPH_VALIDATION_CHECKS).issubset(passed_names)
        assert mock_instantiate.call_args.kwargs == {"preflight_mode": True}
        fake_graph.validate.assert_called_once_with()
        fake_graph.validate_edge_compatibility.assert_called_once_with()

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_graph_structure_failure_uses_graph_check(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_settings = MagicMock()
        fake_graph = MagicMock()
        fake_graph.validate.side_effect = GraphValidationError("bad graph")
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = MagicMock()
        mock_build_graph.return_value = fake_graph

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is False
        assert any(check.name == "schema_compatibility" and not check.passed for check in result.checks)
        fake_graph.validate_edge_compatibility.assert_not_called()

    @patch("elspeth.web.execution.validation.load_settings_from_yaml_string")
    @patch("elspeth.web.execution.validation.instantiate_runtime_plugins")
    @patch("elspeth.web.execution.validation.build_runtime_graph")
    def test_schema_failure_uses_schema_check(
        self,
        mock_build_graph: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        state = _make_state(
            source_options={"path": "/tmp/test_data/blobs/input.csv"},
            outputs=(_make_output({"path": "/tmp/test_data/outputs/out.csv"}),),
        )
        settings = _make_settings(data_dir="/tmp/test_data")
        mock_yaml_gen = MagicMock()
        mock_yaml_gen.generate_yaml.return_value = """
source:
  plugin: csv
  on_success: primary
  options:
    path: /tmp/test_data/blobs/input.csv
    on_validation_failure: discard
sinks:
  primary:
    plugin: csv
    options:
      path: /tmp/test_data/outputs/out.csv
"""
        fake_settings = MagicMock()
        fake_graph = MagicMock()
        fake_graph.validate_edge_compatibility.side_effect = GraphValidationError("schema mismatch")
        mock_load.return_value = fake_settings
        mock_instantiate.return_value = MagicMock()
        mock_build_graph.return_value = fake_graph

        result = validate_pipeline(state, settings, mock_yaml_gen)

        assert result.is_valid is False
        assert _check(result, "graph_structure").passed is True
        assert _check(result, "schema_compatibility").passed is False
```

Also create `tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py`:

```python
"""Persistent guard for runtime-preflight test patch targets."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXECUTION_SERVICE_TEST = PROJECT_ROOT / "tests" / "unit" / "web" / "execution" / "test_service.py"
STALE_RUNTIME_GRAPH_PATCH_TARGETS = (
    "elspeth.web.execution.service.ExecutionGraph",
    "elspeth.web.execution.service.instantiate_plugins_from_config",
)


def test_execution_service_tests_do_not_patch_old_runtime_graph_symbols() -> None:
    text = EXECUTION_SERVICE_TEST.read_text(encoding="utf-8")
    stale_targets = [target for target in STALE_RUNTIME_GRAPH_PATCH_TARGETS if target in text]

    assert stale_targets == [], (
        "Patch runtime graph setup through elspeth.web.execution.preflight, not "
        f"elspeth.web.execution.service: {stale_targets}"
    )
```

Also create `tests/unit/web/execution/test_preflight_side_effects.py`:

```python
"""Runtime preflight must not touch external systems during plugin setup."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode_enabled
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.execution.validation import validate_pipeline


def _forbid_socket_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("runtime preflight plugin instantiation must not open network sockets")

    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket, "getaddrinfo", fail)


def _web_settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=10,
        composer_max_discovery_turns=5,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=60,
    )


def _csv_worker_probe_state(tmp_path: Path) -> CompositionState:
    blobs_dir = tmp_path / "blobs"
    outputs_dir = tmp_path / "outputs"
    blobs_dir.mkdir()
    outputs_dir.mkdir()
    input_path = blobs_dir / "input.csv"
    input_path.write_text("name\nAda\n", encoding="utf-8")
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="primary",
            options={"path": str(input_path), "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="primary",
                plugin="csv",
                options={"path": str(outputs_dir / "out.csv"), "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_preflight_mode_instantiates_external_plugins_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Representative external constructors stay pure in preflight mode.

    Include plugins whose real runtime path creates Azure, Dataverse, OpenAI/
    OpenRouter, RAG, Chroma, or HTTP clients in lifecycle methods. Prefer each
    plugin's probe_config() where it exists; the test is about constructor
    purity, not live credentials.
    """
    pipeline_yaml = _external_plugin_probe_pipeline_yaml(tmp_path)
    settings = load_settings_from_yaml_string(pipeline_yaml)

    _forbid_socket_calls(monkeypatch)
    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)

    assert bundle.source is not None
    assert bundle.sinks


@pytest.mark.asyncio
async def test_run_sync_in_worker_preserves_preflight_mode_for_plugin_constructors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Constructors see preflight mode through the production worker path.

    This pins the runtime contract, not the ContextVar implementation detail:
    validate_pipeline() may run in a ThreadPoolExecutor via run_sync_in_worker(),
    and constructors must still observe preflight mode inside that worker.
    """
    observed: list[tuple[str, bool]] = []
    original_source_init = CSVSource.__init__
    original_sink_init = CSVSink.__init__

    def source_init(self: CSVSource, config: dict[str, Any]) -> None:
        observed.append(("source", plugin_preflight_mode_enabled()))
        original_source_init(self, config)

    def sink_init(self: CSVSink, config: dict[str, Any]) -> None:
        observed.append(("sink", plugin_preflight_mode_enabled()))
        original_sink_init(self, config)

    monkeypatch.setattr(CSVSource, "__init__", source_init)
    monkeypatch.setattr(CSVSink, "__init__", sink_init)

    result = await run_sync_in_worker(
        validate_pipeline,
        _csv_worker_probe_state(tmp_path),
        _web_settings(tmp_path),
        yaml_generator,
    )

    assert result.is_valid is True
    assert observed == [("source", True), ("sink", True)]


def test_runtime_mode_default_does_not_enable_preflight_context(tmp_path: Path) -> None:
    """The normal execution path must remain real runtime mode by default."""
    pipeline_yaml = _minimal_csv_pipeline_yaml(tmp_path)
    settings = load_settings_from_yaml_string(pipeline_yaml)

    bundle = instantiate_plugins_from_config(settings)

    assert bundle.source is not None
```

Implement the two YAML fixture helpers in this file using existing valid test
settings from nearby plugin tests rather than inventing new schema shapes. The
probe pipeline must include at least one representative from each risky family
that is available in this checkout:

- Azure Blob source or sink: assert `AzureAuthConfig.create_blob_service_client()` is not called during preflight.
- Dataverse source or sink: assert `DataverseAuthConfig.create_credential()` and `DataverseClient(...)` are not called during preflight.
- LLM transform: assert `LLMTransform._create_provider()` and `_configure_azure_monitor()` are not called during preflight.
- RAG/chroma/external retrieval or sink if configured in the checkout: assert `chromadb.HttpClient(...)`, provider clients, and health checks are not called during preflight.

If an optional dependency is not installed in the test environment, do not skip
the entire guard. Use the plugin's no-network `probe_config()` or monkeypatch the
external module import with a stub that fails on network/client construction.
The guard is proving that `preflight_mode=True` avoids side effects, not that
Azure/Dataverse/Chroma dependencies are installed.

Keep the two preflight-mode tests separate. The direct-kwarg test isolates
constructor purity for representative external plugins. The worker-path test
pins the production contract: `run_sync_in_worker()` -> `validate_pipeline()` ->
plugin constructors must observe `plugin_preflight_mode_enabled() is True`. The
worker-path test must not wrap the parent coroutine in `plugin_preflight_mode()`
or call `instantiate_plugins_from_config(..., preflight_mode=True)` directly,
because that would miss the executor boundary.

- [ ] **Step 2: Run the failing boundary tests**

Run:

```bash
uv run pytest -q tests/unit/web/execution/test_validation.py::TestValidatePipelineRuntimeCheckBoundaries
uv run pytest -q tests/unit/web/execution/test_preflight_side_effects.py
uv run pytest -q tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py
```

Expected: the validation tests fail until `validation.py` imports and uses `RUNTIME_GRAPH_VALIDATION_CHECKS`, `instantiate_runtime_plugins()`, and `build_runtime_graph()` separately. The direct side-effect test fails until `instantiate_plugins_from_config(..., preflight_mode=True)` exists and representative external constructors are pure under that mode. The worker-path test fails until the production `run_sync_in_worker()` -> `validate_pipeline()` path passes `preflight_mode=True` inside the worker-dispatched plugin-instantiation call. The patch-target guard fails until stale `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config` patches in `tests/unit/web/execution/test_service.py` are migrated. These are the CI bindings that keep composer preflight and runtime graph setup from drifting apart silently.

- [ ] **Step 3: Update validation to use shared helpers but preserve check names**

In `src/elspeth/web/execution/validation.py`, import:

```python
from elspeth.web.execution.preflight import (
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
    RUNTIME_GRAPH_VALIDATION_CHECKS,
    build_runtime_graph,
    instantiate_runtime_plugins,
    resolve_runtime_yaml_paths,
)
```

Bind the check-name constants from named constants, not positional tuple unpacking:

```python
_CHECK_PLUGINS = RUNTIME_CHECK_PLUGIN_INSTANTIATION
_CHECK_GRAPH = RUNTIME_CHECK_GRAPH_STRUCTURE
_CHECK_SCHEMA = RUNTIME_CHECK_SCHEMA_COMPATIBILITY
assert RUNTIME_GRAPH_VALIDATION_CHECKS == (_CHECK_PLUGINS, _CHECK_GRAPH, _CHECK_SCHEMA)
```

Keep the earlier preflight-only checks (`path_allowlist`, `secret_refs`, `semantic_contracts`, `batch_transform_options`, `settings_load`) ahead of these names in `_ALL_CHECKS`.

Replace the plugin/graph/schema section with this shape:

```python
    try:
        bundle = instantiate_runtime_plugins(elspeth_settings, preflight_mode=True)
        checks.append(
            ValidationCheck(
                name=_CHECK_PLUGINS,
                passed=True,
                detail="All plugins instantiated",
            )
        )
    except (PluginNotFoundError, PluginConfigError) as exc:
        comp_type = _infer_component_type_from_plugin_error(exc)
        plugin_name = exc.plugin_name if isinstance(exc, PluginConfigError) else None
        if isinstance(exc, PluginConfigError) and exc.cause is not None and plugin_name is not None:
            detail = f"Invalid configuration for {comp_type} '{plugin_name}': {exc.cause}"
        else:
            detail = str(exc)
        checks.append(ValidationCheck(name=_CHECK_PLUGINS, passed=False, detail=detail))
        errors.append(
            ValidationError(
                component_id=plugin_name,
                component_type=comp_type,
                message=detail,
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_PLUGINS))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    try:
        graph = build_runtime_graph(elspeth_settings, bundle)
        graph.validate()
        checks.append(
            ValidationCheck(
                name=_CHECK_GRAPH,
                passed=True,
                detail="Graph structure is valid",
            )
        )
    except GraphValidationError as exc:
        checks.append(ValidationCheck(name=_CHECK_GRAPH, passed=False, detail=str(exc)))
        errors.append(
            ValidationError(
                component_id=exc.component_id,
                component_type=exc.component_type,
                message=str(exc),
                suggestion=None,
            )
        )
        checks.extend(_skipped_checks(_CHECK_GRAPH))
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )

    try:
        graph.validate_edge_compatibility()
        checks.append(
            ValidationCheck(
                name=_CHECK_SCHEMA,
                passed=True,
                detail="All edge schemas compatible",
            )
        )
    except GraphValidationError as exc:
        checks.append(ValidationCheck(name=_CHECK_SCHEMA, passed=False, detail=str(exc)))
        errors.append(
            ValidationError(
                component_id=exc.component_id,
                component_type=exc.component_type,
                message=str(exc),
                suggestion=None,
            )
        )
        return ValidationResult(
            is_valid=False,
            checks=checks,
            errors=errors,
            semantic_contracts=serialize_semantic_contracts(semantic_contracts),
        )
```

- [ ] **Step 3A: Add plugin preflight mode without weakening runtime execution**

Create `src/elspeth/plugins/infrastructure/preflight.py`:

```python
"""Process-local plugin-instantiation preflight mode."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_PLUGIN_PREFLIGHT_MODE: ContextVar[bool] = ContextVar("elspeth_plugin_preflight_mode", default=False)


def plugin_preflight_mode_enabled() -> bool:
    """Return True while plugins are being instantiated for runtime preflight."""
    return _PLUGIN_PREFLIGHT_MODE.get()


@contextmanager
def plugin_preflight_mode(enabled: bool) -> Iterator[None]:
    token = _PLUGIN_PREFLIGHT_MODE.set(enabled)
    try:
        yield
    finally:
        _PLUGIN_PREFLIGHT_MODE.reset(token)
```

Modify `src/elspeth/cli_helpers.py`:

```python
def instantiate_plugins_from_config(
    config: "ElspethSettings",
    *,
    preflight_mode: bool = False,
) -> PluginBundle:
    ...
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode

    with plugin_preflight_mode(preflight_mode):
        # existing source/transform/aggregation/sink instantiation logic
```

Keep the default `preflight_mode=False` so CLI and execution semantics do not
change. Do not implement this by passing an extra keyword to plugin constructors:
existing constructors have a closed `__init__(config)` contract, and catching
`TypeError` to retry would mask real plugin bugs. The contextvar is an explicit
mechanical affordance for constructors that must defer external setup during
preflight while preserving their public constructor signature.

Threading contract: set the plugin preflight context inside
`instantiate_plugins_from_config(config, preflight_mode=True)` while it is
running in the worker thread. Do not set `plugin_preflight_mode(True)` in the
async caller before `run_sync_in_worker()` and assume the `ContextVar` will
propagate across executor implementations. The persistent
`test_run_sync_in_worker_preserves_preflight_mode_for_plugin_constructors()`
must exercise the production path and is the regression guard for this boundary.

Audit every plugin constructor matched by:

```bash
rg -n "__init__\\(self, config" src/elspeth/plugins
```

Required audit outcomes:

- Config parsing, schema construction, declared field initialization, and pure template compilation stay in `__init__`.
- Network/client/credential/provider setup stays in `on_start()`, first real operation, or an already-lazy helper.
- Known hot spots to verify explicitly: `src/elspeth/plugins/infrastructure/azure_auth.py::AzureAuthConfig.create_blob_service_client`, `src/elspeth/plugins/infrastructure/clients/dataverse.py::DataverseAuthConfig.create_credential`, `DataverseClient(...)`, `src/elspeth/plugins/transforms/llm/transform.py::_create_provider`, `_configure_azure_monitor`, `src/elspeth/plugins/sinks/chroma_sink.py::on_start`, and retrieval provider client factories.
- If a constructor currently does side-effectful setup, guard only that setup with `plugin_preflight_mode_enabled()` and defer it to the existing lifecycle/operation path. Do not skip validation or schema setup in preflight mode.

- [ ] **Step 4: Update execution service to use the all-in-one runtime helper**

In `src/elspeth/web/execution/service.py`, import:

```python
from elspeth.web.execution.preflight import build_validated_runtime_graph, resolve_runtime_yaml_paths
```

Replace the `_run_pipeline()` plugin/graph setup block with:

```python
            runtime_graph = build_validated_runtime_graph(settings)
            bundle = runtime_graph.plugin_bundle
            graph = runtime_graph.graph
```

Remove now-unused imports of `instantiate_plugins_from_config` and `ExecutionGraph` from `execution/service.py`.

- [ ] **Step 5: Run validation and execution service tests**

Run:

```bash
uv run pytest -q tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_preflight_side_effects.py tests/unit/web/execution/test_service.py tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py
```

Expected: pass. Required patch-target updates are:

- Tests that patch `elspeth.web.execution.service.ExecutionGraph` should patch `elspeth.web.execution.preflight.ExecutionGraph`.
- Tests that patch `elspeth.web.execution.service.instantiate_plugins_from_config` should patch `elspeth.web.execution.preflight.instantiate_plugins_from_config`.
- Tests that verify `validate_pipeline()` orchestration should patch the new symbols imported by `elspeth.web.execution.validation`, not the execution-service module.

Mechanically migrate these verified `tests/unit/web/execution/test_service.py` patch-site pairs first:

- Lines 345/346: `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config`.
- Lines 412/413: `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config`.
- Lines 638/639: `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config`.
- Lines 693/694: `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config`.
- Lines 755/756: `elspeth.web.execution.service.ExecutionGraph` and `elspeth.web.execution.service.instantiate_plugins_from_config`.

The persistent guard in `tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py` replaces the old one-shot `rg` check. After migrating the listed pairs, run:

```bash
uv run pytest -q tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/preflight.py src/elspeth/cli_helpers.py src/elspeth/web/execution/preflight.py src/elspeth/web/execution/validation.py src/elspeth/web/execution/service.py tests/unit/web/execution/test_preflight_side_effects.py tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py
git commit -m "refactor(web): share runtime graph setup for validation and execution"
```

---

### Task 3: Expose Bounded Cached Runtime Preflight Through `preview_pipeline`

**Files:**
- Modify: `src/elspeth/web/config.py`
- Modify: `src/elspeth/web/composer/tools.py`
- Modify: `src/elspeth/web/composer/protocol.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `scripts/cicd/enforce_composer_catch_order.py`
- Test: `tests/unit/web/composer/test_tools.py`
- Test: `tests/unit/web/composer/test_service.py`
- Test: `tests/unit/scripts/cicd/test_enforce_composer_catch_order.py`

- [ ] **Step 1: Add failing preview test**

Append to `tests/unit/web/composer/test_tools.py` inside `TestPreviewPipeline`:

```python
    def test_preview_pipeline_surfaces_runtime_preflight_failure(self) -> None:
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        catalog = _mock_catalog()
        runtime_preflight = MagicMock(
            return_value=ValidationResult(
                is_valid=False,
                checks=[
                    ValidationCheck(
                        name="settings_load",
                        passed=False,
                        detail="Forbidden name: 'end_of_source'",
                    )
                ],
                errors=[
                    ValidationError(
                        component_id="agg1",
                        component_type="aggregation",
                        message="Forbidden name: 'end_of_source'",
                        suggestion="Omit trigger for end-of-source-only aggregation.",
                    )
                ],
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            catalog,
            data_dir="/data",
            runtime_preflight=runtime_preflight,
        )

        assert result.success is True
        assert result.runtime_preflight is not None
        assert result.data["authoring_validation"]["is_valid"] is True
        assert result.data["runtime_preflight"]["is_valid"] is False
        assert result.data["is_valid"] is False
        assert result.data["runtime_preflight"]["errors"][0]["message"] == "Forbidden name: 'end_of_source'"
        runtime_preflight.assert_called_once_with(state)

    def test_preview_pipeline_without_runtime_preflight_preserves_authoring_validation(self) -> None:
        state = _empty_state().with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )

        result = execute_tool(
            "preview_pipeline",
            {},
            state,
            _mock_catalog(),
            data_dir="/data",
            runtime_preflight=None,
        )

        assert result.success is True
        assert result.runtime_preflight is None
        assert result.data["authoring_validation"]["is_valid"] is True
        assert result.data["runtime_preflight"] is None
        assert result.data["is_valid"] is True
```

Add imports if needed:

```python
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationResult
```

Append to `tests/unit/web/composer/test_service.py`:

```python
class TestComposerRuntimePreflightCacheAndTimeout:
    @pytest.mark.asyncio
    async def test_runtime_preflight_cache_reuses_same_state_version_and_settings_hash(self) -> None:
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=preflight) as mock_preflight:
            first = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )
            second = await service._cached_runtime_preflight(
                state,
                user_id="user-1",
                cache=cache,
                initial_version=state.version,
                session_scope="session:test",
            )

        assert first is preflight
        assert second is preflight
        mock_preflight.assert_called_once_with(state, "user-1")

    @pytest.mark.asyncio
    async def test_runtime_preflight_timeout_is_cached_for_compose_call(self) -> None:
        settings = _make_settings(composer_runtime_preflight_timeout_seconds=0.01)
        service = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
        state = _empty_state()
        cache = service._new_runtime_preflight_cache()
        started = threading.Event()
        release = threading.Event()

        def slow_preflight(candidate: CompositionState, user_id: str | None) -> ValidationResult:
            started.set()
            release.wait(timeout=30)
            return ValidationResult(is_valid=True, checks=[], errors=[])

        try:
            with patch.object(service, "_runtime_preflight", side_effect=slow_preflight) as mock_preflight:
                with pytest.raises(ComposerRuntimePreflightError) as first:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )
                assert first.value.exc_class == "TimeoutError"
                assert started.is_set()

                with pytest.raises(ComposerRuntimePreflightError) as second:
                    await service._cached_runtime_preflight(
                        state,
                        user_id="user-1",
                        cache=cache,
                        initial_version=state.version - 1,
                        session_scope="session:test",
                    )

                assert second.value.exc_class == "TimeoutError"
                mock_preflight.assert_called_once()
        finally:
            release.set()

    def test_runtime_preflight_settings_hash_is_non_secret(self) -> None:
        class FakeSettings:
            data_dir = Path("/tmp/elspeth-data")
            landscape_passphrase = "SECRET_CANARY_SHOULD_NOT_APPEAR"

        digest = runtime_preflight_settings_hash(FakeSettings())

        assert "SECRET_CANARY" not in digest
        assert len(digest) == 64
```

Create `tests/unit/web/execution/test_runtime_preflight_coordinator.py`:

```python
"""Runtime preflight in-flight coordination tests."""

from __future__ import annotations

import asyncio

import pytest

from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import ValidationResult


@pytest.mark.asyncio
async def test_coordinator_deduplicates_concurrent_same_session_state_settings() -> None:
    coordinator = RuntimePreflightCoordinator()
    key = RuntimePreflightKey(
        session_scope="session:abc123",
        state_version=7,
        settings_hash="settings-hash",
    )
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    expected = ValidationResult(is_valid=True, checks=[], errors=[])

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return expected

    first_task = asyncio.create_task(coordinator.run(key, worker))
    await started.wait()
    second_task = asyncio.create_task(coordinator.run(key, worker))

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is expected
    assert second is expected
    assert calls == 1


@pytest.mark.asyncio
async def test_coordinator_deduplicates_concurrent_failure_for_same_key() -> None:
    coordinator = RuntimePreflightCoordinator()
    key = RuntimePreflightKey(
        session_scope="session:abc123",
        state_version=7,
        settings_hash="settings-hash",
    )
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    original = RuntimeError("constructor failed")

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        raise original

    first_task = asyncio.create_task(coordinator.run(key, worker))
    await started.wait()
    second_task = asyncio.create_task(coordinator.run(key, worker))

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert isinstance(first, RuntimePreflightFailure)
    assert isinstance(second, RuntimePreflightFailure)
    assert first.original_exc is original
    assert second.original_exc is original
    assert calls == 1


@pytest.mark.asyncio
async def test_coordinator_does_not_share_different_session_scopes() -> None:
    coordinator = RuntimePreflightCoordinator()
    calls = 0

    async def worker() -> ValidationResult:
        nonlocal calls
        calls += 1
        return ValidationResult(is_valid=True, checks=[], errors=[])

    await asyncio.gather(
        coordinator.run(RuntimePreflightKey("session:http", 1, "settings"), worker),
        coordinator.run(RuntimePreflightKey("session:mcp", 1, "settings"), worker),
    )

    assert calls == 2
```

Append to `tests/unit/composer_mcp/test_server.py`:

```python
@pytest.mark.asyncio
async def test_mcp_preview_runtime_preflight_joins_shared_session_inflight() -> None:
    from elspeth.composer_mcp.server import _mcp_preview_runtime_preflight
    from elspeth.web.execution.runtime_preflight import RuntimePreflightCoordinator
    from elspeth.web.execution.schemas import ValidationResult

    coordinator = RuntimePreflightCoordinator()
    state = _valid_state_with_no_edge_contracts()
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    expected = ValidationResult(is_valid=True, checks=[], errors=[])

    async def run_preflight(candidate: CompositionState) -> ValidationResult:
        nonlocal calls
        assert candidate is state
        calls += 1
        started.set()
        await release.wait()
        return expected

    first_task = asyncio.create_task(
        _mcp_preview_runtime_preflight(
            state,
            coordinator=coordinator,
            session_scope="session:shared-web-session",
            settings_hash="settings-hash",
            timeout_seconds=1.0,
            run_preflight=run_preflight,
        )
    )
    await started.wait()
    second_task = asyncio.create_task(
        _mcp_preview_runtime_preflight(
            state,
            coordinator=coordinator,
            session_scope="session:shared-web-session",
            settings_hash="settings-hash",
            timeout_seconds=1.0,
            run_preflight=run_preflight,
        )
    )

    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is expected
    assert second is expected
    assert calls == 1
```

Add imports if needed:

```python
import asyncio
import threading
from pathlib import Path

from elspeth.web.composer.protocol import ComposerRuntimePreflightError
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run pytest -q tests/unit/web/composer/test_tools.py::TestPreviewPipeline::test_preview_pipeline_surfaces_runtime_preflight_failure tests/unit/web/composer/test_tools.py::TestPreviewPipeline::test_preview_pipeline_without_runtime_preflight_preserves_authoring_validation
uv run pytest -q tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightCacheAndTimeout
uv run pytest -q tests/unit/web/execution/test_runtime_preflight_coordinator.py
uv run pytest -q tests/unit/composer_mcp/test_server.py::test_mcp_preview_runtime_preflight_joins_shared_session_inflight
uv run pytest -q tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
```

Expected: fail because `execute_tool()` does not accept `runtime_preflight`, `ToolResult` has no `runtime_preflight` field, the runtime-preflight coordinator module does not exist, composer MCP has no runtime-preflight path, composer service has no scoped runtime-preflight cache or preflight timeout setting yet, and the catch-order registry does not know the new exception class yet.

- [ ] **Step 2A: Add transport-neutral runtime preflight coordinator**

Create `src/elspeth/web/execution/runtime_preflight.py`:

```python
"""Async coordination for runtime-equivalent composer preflight.

This module is transport-neutral L3 infrastructure. HTTP composer and
composer MCP can share the same process-local coordinator when they share a
logical session. Standalone MCP processes cannot share this lock with the web
process; cross-process safety comes from side-effect-free preflight mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from elspeth.web.execution.schemas import ValidationResult


@dataclass(frozen=True, slots=True)
class RuntimePreflightKey:
    session_scope: str
    state_version: int
    settings_hash: str


@dataclass(frozen=True, slots=True)
class RuntimePreflightFailure:
    original_exc: Exception


RuntimePreflightEntry = ValidationResult | RuntimePreflightFailure
RuntimePreflightWorker = Callable[[], Awaitable[ValidationResult]]


class RuntimePreflightCoordinator:
    """Deduplicate in-flight runtime preflight for one Python process."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight: dict[RuntimePreflightKey, asyncio.Task[RuntimePreflightEntry]] = {}

    async def run(
        self,
        key: RuntimePreflightKey,
        worker: RuntimePreflightWorker,
    ) -> RuntimePreflightEntry:
        async with self._lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._capture(worker))
                self._inflight[key] = task

        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._inflight.get(key) is task:
                        self._inflight.pop(key, None)

    async def _capture(self, worker: RuntimePreflightWorker) -> RuntimePreflightEntry:
        try:
            return await worker()
        except Exception as exc:
            return RuntimePreflightFailure(exc)
```

This coordinator intentionally does not keep a completed-result cache. The
per-compose cache in `ComposerServiceImpl` owns reuse within one composer call;
the coordinator only joins concurrent in-flight work across transports.

- [ ] **Step 3: Add `ToolResult.runtime_preflight` and payload helper**

In `src/elspeth/web/composer/tools.py`, import `ValidationResult`:

```python
from elspeth.web.execution.schemas import ValidationResult
```

Add type alias near `ToolHandler`:

```python
RuntimePreflight = Callable[[CompositionState], ValidationResult]
```

Extend `ToolResult`:

```python
    runtime_preflight: ValidationResult | None = None
```

Update `to_dict()` so runtime preflight is visible to the LLM:

```python
        if self.runtime_preflight is not None:
            result["runtime_preflight"] = self.runtime_preflight.model_dump()
```

Add helper:

```python
def _authoring_validation_payload(validation: ValidationSummary) -> dict[str, Any]:
    return {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
        "edge_contracts": [ec.to_dict() for ec in validation.edge_contracts],
        "semantic_contracts": _semantic_contracts_payload(validation.semantic_contracts),
    }
```

- [ ] **Step 4: Thread runtime preflight into preview only**

Change `_execute_preview_pipeline()` signature:

```python
def _execute_preview_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    runtime_preflight: RuntimePreflight | None = None,
) -> ToolResult:
```

Use this summary shape:

```python
    validation = state.validate()
    authoring_payload = _authoring_validation_payload(validation)
    runtime_result = runtime_preflight(state) if runtime_preflight is not None else None

    summary: dict[str, Any] = {
        "is_valid": validation.is_valid if runtime_result is None else validation.is_valid and runtime_result.is_valid,
        "errors": authoring_payload["errors"],
        "warnings": authoring_payload["warnings"],
        "suggestions": authoring_payload["suggestions"],
        "edge_contracts": authoring_payload["edge_contracts"],
        "semantic_contracts": authoring_payload["semantic_contracts"],
        "authoring_validation": authoring_payload,
        "runtime_preflight": runtime_result.model_dump() if runtime_result is not None else None,
        "source": None,
        "node_count": len(state.nodes),
        "output_count": len(state.outputs),
        "nodes": [{"id": n.id, "node_type": n.node_type, "plugin": n.plugin} for n in state.nodes],
        "outputs": [{"name": o.name, "plugin": o.plugin} for o in state.outputs],
    }
```

Return:

```python
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=summary,
        runtime_preflight=runtime_result,
    )
```

- [ ] **Step 5: Add `runtime_preflight` to `execute_tool()`**

Change the `execute_tool()` signature:

```python
    runtime_preflight: RuntimePreflight | None = None,
```

Add this special case before `diff_pipeline`:

```python
    if tool_name == "preview_pipeline":
        return _execute_preview_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            runtime_preflight=runtime_preflight,
        )
```

- [ ] **Step 5A: Add a dedicated runtime-preflight timeout setting**

Modify `src/elspeth/web/config.py`:

```python
    composer_runtime_preflight_timeout_seconds: float = Field(default=5.0, gt=0)
```

Place it next to `composer_timeout_seconds`. This is deliberately separate from
the overall compose-loop deadline: a runtime preflight worker can hang in plugin
construction even when the compose loop has remaining time.

Modify `ComposerSettings` in `src/elspeth/web/composer/protocol.py`:

```python
    @property
    def composer_runtime_preflight_timeout_seconds(self) -> float: ...
```

Also introduce `ComposerRuntimePreflightError` in
`src/elspeth/web/composer/protocol.py` in this task, because the cache/timeout
helper raises it before Task 4 adds final-response gating:

```python
from typing import ClassVar


class ComposerRuntimePreflightError(ComposerServiceError):
    """Unexpected internal failure while running composer runtime preflight."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"original_exc", "partial_state", "exc_class"})

    def __init__(self, *, original_exc: Exception, partial_state: CompositionState | None) -> None:
        super().__init__("Composer runtime preflight failed internally.")
        self.original_exc = original_exc
        self.partial_state = partial_state
        self.exc_class = type(original_exc).__name__

    def __setattr__(self, name: str, value: object) -> None:
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(f"{type(self).__name__}.{name} is frozen after construction")
        super().__setattr__(name, value)

    @classmethod
    def capture(
        cls,
        exc: Exception,
        *,
        state: CompositionState,
        initial_version: int,
    ) -> ComposerRuntimePreflightError:
        partial = state if state.version > initial_version else None
        return cls(original_exc=exc, partial_state=partial)
```

Because this task introduces the exception class, update catch-order enforcement
in the same commit instead of waiting for final-response gating. Apply the
registry/test changes described in Task 4 Step 4 now; Task 4 Step 4 becomes a
verification pass if they are already present.

Do not add this field to `ValidationSettings`; validation itself is synchronous
and reusable outside composer. The async boundary that calls validation owns the
timeout.

- [ ] **Step 6: Build the runtime preflight callback in composer service**

In `src/elspeth/web/composer/service.py`, import:

```python
import asyncio
from dataclasses import dataclass
from typing import NoReturn

from elspeth.web.composer import yaml_generator
from elspeth.web.composer.protocol import ComposerRuntimePreflightError
from elspeth.web.composer.tools import RuntimePreflight
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightEntry,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.execution.validation import validate_pipeline
```

Extend `ComposerServiceImpl.__init__()`:

```python
        runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
```

Store settings and the coordinator:

```python
        self._settings = settings
        self._runtime_preflight_timeout_seconds = settings.composer_runtime_preflight_timeout_seconds
        self._runtime_preflight_coordinator = runtime_preflight_coordinator or RuntimePreflightCoordinator()
```

Add cache types near the discovery-cache helper types:

```python
_RuntimePreflightCache = dict[RuntimePreflightKey, RuntimePreflightEntry]
```

Add:

```python
    def _runtime_preflight(self, state: CompositionState, user_id: str | None) -> ValidationResult:
        return validate_pipeline(
            state,
            self._settings,
            yaml_generator,
            secret_service=self._secret_service,
            user_id=user_id,
        )

    def _new_runtime_preflight_cache(self) -> _RuntimePreflightCache:
        return {}

    def _raise_cached_runtime_preflight_failure(
        self,
        failure: RuntimePreflightFailure,
        *,
        state: CompositionState,
        initial_version: int,
    ) -> NoReturn:
        raise ComposerRuntimePreflightError.capture(
            failure.original_exc,
            state=state,
            initial_version=initial_version,
        ) from failure.original_exc

    async def _cached_runtime_preflight(
        self,
        state: CompositionState,
        *,
        user_id: str | None,
        cache: _RuntimePreflightCache,
        initial_version: int,
        session_scope: str,
    ) -> ValidationResult:
        key = RuntimePreflightKey(
            session_scope=session_scope,
            state_version=state.version,
            settings_hash=runtime_preflight_settings_hash(self._settings),
        )
        cached = cache.get(key)
        if isinstance(cached, ValidationResult):
            return cached
        if isinstance(cached, RuntimePreflightFailure):
            self._raise_cached_runtime_preflight_failure(
                cached,
                state=state,
                initial_version=initial_version,
            )

        async def worker() -> ValidationResult:
            return await asyncio.wait_for(
                run_sync_in_worker(self._runtime_preflight, state, user_id),
                timeout=self._runtime_preflight_timeout_seconds,
            )

        entry = await self._runtime_preflight_coordinator.run(key, worker)
        cache[key] = entry
        if isinstance(entry, RuntimePreflightFailure):
            self._raise_cached_runtime_preflight_failure(
                entry,
                state=state,
                initial_version=initial_version,
            )
        return entry
```

Important: the timeout above bounds how long the compose coroutine waits. It
does not kill a running Python thread. That is why timeout/internal-failure
entries are cached for this compose call; after one timeout, later
`preview_pipeline` calls and finalization for the same `(session_scope,
state.version, settings_hash)` must re-raise the cached failure instead of
scheduling another worker. The coordinator additionally joins concurrent HTTP
and embedded MCP preview requests for the same key while the first worker is
still running.

Initialize before the compose loop:

```python
        runtime_preflight_cache = self._new_runtime_preflight_cache()
        last_runtime_preflight: ValidationResult | None = None
        session_scope = f"session:{session_id}" if session_id is not None else "session:unsaved"
```

Before calling `execute_tool()` for each tool call, precompute preview runtime
preflight outside the general side-effectful tool worker:

```python
                runtime_preflight_callback: RuntimePreflight | None = None
                if tool_name == "preview_pipeline":
                    preview_preflight = await self._cached_runtime_preflight(
                        state,
                        user_id=user_id,
                        cache=runtime_preflight_cache,
                        initial_version=initial_version,
                        session_scope=session_scope,
                    )
                    runtime_preflight_callback = lambda _state, result=preview_preflight: result
```

Then pass the cheap callback into `execute_tool()`:

```python
                        runtime_preflight=runtime_preflight_callback,
```

This keeps `execute_tool()` synchronous and preserves the direct unit-test API,
but the expensive runtime preflight is now bounded and cached in the async
compose loop. Do not run `validate_pipeline()` from inside `execute_tool()` in
the service path.

Track preview preflight after tool execution:

```python
                last_runtime_preflight = result.runtime_preflight or last_runtime_preflight
```

- [ ] **Step 6A: Thread the coordinator through web app and composer MCP**

In `src/elspeth/web/app.py`, create one coordinator per web process and pass it
into the composer service:

```python
from elspeth.web.execution.runtime_preflight import RuntimePreflightCoordinator
```

At app construction, before `ComposerServiceImpl(...)`:

```python
    runtime_preflight_coordinator = RuntimePreflightCoordinator()
    app.state.runtime_preflight_coordinator = runtime_preflight_coordinator
```

When constructing `ComposerServiceImpl`, pass:

```python
        runtime_preflight_coordinator=runtime_preflight_coordinator,
```

In `src/elspeth/composer_mcp/server.py`, add imports:

```python
import asyncio
from collections.abc import Awaitable, Callable

from elspeth.web.composer.tools import RuntimePreflight
from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import ValidationResult
```

Add MCP helper types and helper near `_dispatch_tool()`:

```python
McpRuntimePreflight = Callable[[CompositionState], Awaitable[ValidationResult]]
SessionScopeProvider = Callable[[], str]


async def _mcp_preview_runtime_preflight(
    state: CompositionState,
    *,
    coordinator: RuntimePreflightCoordinator,
    session_scope: str,
    settings_hash: str,
    timeout_seconds: float,
    run_preflight: McpRuntimePreflight,
) -> ValidationResult:
    key = RuntimePreflightKey(
        session_scope=session_scope,
        state_version=state.version,
        settings_hash=settings_hash,
    )

    async def worker() -> ValidationResult:
        return await asyncio.wait_for(run_preflight(state), timeout=timeout_seconds)

    entry = await coordinator.run(key, worker)
    if isinstance(entry, RuntimePreflightFailure):
        raise entry.original_exc
    return entry
```

Extend `_dispatch_tool()`:

```python
    runtime_preflight: RuntimePreflight | None = None,
```

and pass it through:

```python
        result = execute_tool(
            tool_name,
            arguments,
            state,
            catalog,
            data_dir=None,
            baseline=baseline,
            runtime_preflight=runtime_preflight,
        )
```

Extend `create_server()`:

```python
    runtime_preflight: McpRuntimePreflight | None = None,
    runtime_preflight_settings_hash: str | None = None,
    runtime_preflight_timeout_seconds: float = 5.0,
    runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
    session_scope_provider: SessionScopeProvider | None = None,
```

Inside `create_server()`:

```python
    coordinator = runtime_preflight_coordinator or RuntimePreflightCoordinator()
    session_id_ref: list[str | None] = [None]

    def current_session_scope() -> str:
        if session_scope_provider is not None:
            return session_scope_provider()
        session_id = session_id_ref[0] or "unsaved"
        return f"composer-mcp:{scratch_dir.resolve()}:{session_id}"
```

In `call_tool()`, before `_dispatch_tool()`:

```python
            runtime_preflight_callback: RuntimePreflight | None = None
            if name == "preview_pipeline" and runtime_preflight is not None:
                if runtime_preflight_settings_hash is None:
                    raise ValueError("runtime_preflight_settings_hash is required when runtime_preflight is configured")
                preview_preflight = await _mcp_preview_runtime_preflight(
                    state_ref[0],
                    coordinator=coordinator,
                    session_scope=current_session_scope(),
                    settings_hash=runtime_preflight_settings_hash,
                    timeout_seconds=runtime_preflight_timeout_seconds,
                    run_preflight=runtime_preflight,
                )
                runtime_preflight_callback = lambda _state, result=preview_preflight: result
```

Then pass `runtime_preflight=runtime_preflight_callback` into `_dispatch_tool()`.
After successful `new_session` or `load_session`, update `session_id_ref[0]`
from `result["data"]["session_id"]`.

Transport contract:

- Embedded/in-process MCP that targets a web session must pass the same
  `RuntimePreflightCoordinator` as HTTP and a `session_scope_provider` returning
  `f"session:{session_id}"`.
- Standalone MCP uses `composer-mcp:{scratch_dir}:{session_id}`. It cannot
  single-flight with the web process, so it relies on `preflight_mode=True`
  purity for cross-process safety.
- Do not run `validate_pipeline()` inside synchronous `_dispatch_tool()`;
  precompute the runtime result in async `call_tool()` and pass the cheap
  callback into `execute_tool()`, mirroring the HTTP composer service path.

- [ ] **Step 7: Run preview tests**

Run:

```bash
uv run pytest -q tests/unit/web/composer/test_tools.py::TestPreviewPipeline
uv run pytest -q tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightCacheAndTimeout
uv run pytest -q tests/unit/web/execution/test_runtime_preflight_coordinator.py
uv run pytest -q tests/unit/composer_mcp/test_server.py::test_mcp_preview_runtime_preflight_joins_shared_session_inflight
uv run pytest -q tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/config.py src/elspeth/web/app.py src/elspeth/web/execution/runtime_preflight.py src/elspeth/web/composer/protocol.py src/elspeth/web/composer/tools.py src/elspeth/web/composer/service.py src/elspeth/composer_mcp/server.py scripts/cicd/enforce_composer_catch_order.py tests/unit/web/execution/test_runtime_preflight_coordinator.py tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_service.py tests/unit/composer_mcp/test_server.py tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
git commit -m "feat(composer): expose runtime preflight in pipeline preview"
```

---

### Task 4: Gate Final Composer Responses And Preserve Raw Model Text

**Files:**
- Modify: `src/elspeth/web/composer/protocol.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `scripts/cicd/enforce_composer_catch_order.py`
- Test: `tests/unit/web/composer/test_service.py`
- Test: `tests/unit/scripts/cicd/test_enforce_composer_catch_order.py`

Precondition: complete and commit Task 3 before starting this task. `ComposerServiceImpl.__init__()` does not currently store `self._settings`; Task 3 Step 6 adds that field plus the compose-scoped runtime-preflight cache/timeout helpers, and Task 4's finalization path depends on them.

- [ ] **Step 1: Add failing final-gate tests**

Append to `tests/unit/web/composer/test_service.py`:

```python
class TestComposerRuntimePreflightFinalGate:
    @pytest.mark.asyncio
    async def test_changed_state_completion_is_replaced_when_runtime_preflight_fails(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        ).with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/data/outputs/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        changed_state = replace(changed_state, version=state.version + 1)

        llm_response = _make_llm_response(content="The pipeline is complete and valid.")
        failed_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion="Omit trigger for end-of-source-only aggregation.",
                )
            ],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = llm_response
            with patch.object(service, "_runtime_preflight", return_value=failed_preflight) as mock_preflight:
                result = await service._finalize_no_tool_response(
                    content="The pipeline is complete and valid.",
                    state=changed_state,
                    initial_version=state.version,
                    user_id="user-1",
                    last_runtime_preflight=None,
                    runtime_preflight_cache=service._new_runtime_preflight_cache(),
                    session_scope="session:test",
                )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is failed_preflight
        mock_preflight.assert_called_once_with(changed_state, "user-1")

    @pytest.mark.asyncio
    async def test_unchanged_text_without_preview_does_not_run_preflight(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="I can help with that.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "I can help with that."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is None
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchanged_state_reuses_preview_preflight_without_rerun(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        preview_preflight = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Forbidden name: 'end_of_source'",
                )
            ],
            errors=[
                ValidationError(
                    component_id="agg1",
                    component_type="aggregation",
                    message="Forbidden name: 'end_of_source'",
                    suggestion=None,
                )
            ],
        )

        with patch.object(service, "_runtime_preflight") as mock_preflight:
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=preview_preflight,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message != "The pipeline is complete and valid."
        assert result.raw_assistant_content == "The pipeline is complete and valid."
        assert result.runtime_preflight is preview_preflight
        mock_preflight.assert_not_called()

    @pytest.mark.asyncio
    async def test_passing_preflight_preserves_original_message_verbatim(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)
        passed_preflight = ValidationResult(is_valid=True, checks=[], errors=[])

        with patch.object(service, "_runtime_preflight", return_value=passed_preflight):
            result = await service._finalize_no_tool_response(
                content="The pipeline is complete and valid.",
                state=changed_state,
                initial_version=state.version,
                user_id="user-1",
                last_runtime_preflight=None,
                runtime_preflight_cache=service._new_runtime_preflight_cache(),
                session_scope="session:test",
            )

        assert result.message == "The pipeline is complete and valid."
        assert result.raw_assistant_content is None
        assert result.runtime_preflight is passed_preflight

    def test_runtime_preflight_failure_message_uses_failed_check_when_errors_empty(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Graph has no path from source to sink",
                )
            ],
            errors=[],
        )

        message = ComposerServiceImpl._runtime_preflight_failure_message(result)

        assert "runtime preflight failed" in message
        assert "Graph has no path from source to sink" in message

    def test_runtime_preflight_failure_message_has_bare_fallback(self) -> None:
        result = ValidationResult(is_valid=False, checks=[], errors=[])

        message = ComposerServiceImpl._runtime_preflight_failure_message(result)

        assert message == "I cannot mark this pipeline complete yet because runtime preflight failed."

    @pytest.mark.asyncio
    async def test_unexpected_preflight_exception_preserves_partial_state(self) -> None:
        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()
        changed_state = replace(state, version=state.version + 1)

        with patch.object(service, "_runtime_preflight", side_effect=RuntimeError("boom")):
            with pytest.raises(ComposerRuntimePreflightError) as exc_info:
                await service._finalize_no_tool_response(
                    content="The pipeline is complete.",
                    state=changed_state,
                    initial_version=state.version,
                    user_id="user-1",
                    last_runtime_preflight=None,
                    runtime_preflight_cache=service._new_runtime_preflight_cache(),
                    session_scope="session:test",
                )

        assert exc_info.value.partial_state == changed_state
        assert exc_info.value.exc_class == "RuntimeError"

    def test_runtime_preflight_error_original_exception_is_frozen(self) -> None:
        original = RuntimeError("boom")
        error = ComposerRuntimePreflightError(original_exc=original, partial_state=None)

        with pytest.raises(AttributeError):
            error.original_exc = RuntimeError("replacement")
```

Add imports if not already present:

```python
from dataclasses import replace
from elspeth.web.composer.protocol import ComposerRuntimePreflightError
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationResult
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
uv run pytest -q tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightFinalGate
```

Expected: fail because `ComposerResult.raw_assistant_content` and deterministic final gating do not exist yet. If `ComposerRuntimePreflightError` is still missing, Task 3 was incomplete; add it before proceeding.

- [ ] **Step 3: Extend composer protocol**

Modify `src/elspeth/web/composer/protocol.py`:

```python
from elspeth.web.execution.schemas import ValidationResult
```

Update `ComposerResult`:

```python
@dataclass(frozen=True, slots=True)
class ComposerResult:
    """Result of a compose() call."""

    message: str
    state: CompositionState
    runtime_preflight: ValidationResult | None = None
    raw_assistant_content: str | None = None
```

Task 3 should already have added `ComposerRuntimePreflightError` for the
cache/timeout helper. If it was not added there, add it now; otherwise verify it
still has exactly this shape:

```python
class ComposerRuntimePreflightError(ComposerServiceError):
    """Unexpected internal failure while running final composer preflight."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"original_exc", "partial_state", "exc_class"})

    def __init__(self, *, original_exc: Exception, partial_state: CompositionState | None) -> None:
        super().__init__("Composer runtime preflight failed internally.")
        self.original_exc = original_exc
        self.partial_state = partial_state
        self.exc_class = type(original_exc).__name__

    def __setattr__(self, name: str, value: object) -> None:
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(f"{type(self).__name__}.{name} is frozen after construction")
        super().__setattr__(name, value)

    @classmethod
    def capture(
        cls,
        exc: Exception,
        *,
        state: CompositionState,
        initial_version: int,
    ) -> ComposerRuntimePreflightError:
        partial = state if state.version > initial_version else None
        return cls(original_exc=exc, partial_state=partial)
```

The `_FROZEN_ATTRS` set intentionally matches `ComposerPluginCrashError`: the captured `original_exc`, `partial_state`, and `exc_class` are provenance-bearing fields and must not be rewritten after construction. `original_exc` is `Exception`, not `BaseException`, because runtime-preflight capture sites handle ordinary `Exception` subclasses such as `TimeoutError`; do not widen it to `BaseException`.

- [ ] **Step 4: Extend composer catch-order CI enforcement**

Task 3 introduces `ComposerRuntimePreflightError`, so this registry/test update
should already be present. If it is present, treat this step as a verification
step and only run the test below. If it is missing, add it now before editing
route catch sites.

Modify `scripts/cicd/enforce_composer_catch_order.py`:

```python
_SUBCLASS_TO_SUPERCLASSES: dict[str, frozenset[str]] = {
    "ComposerPluginCrashError": frozenset({"ComposerServiceError"}),
    "ComposerConvergenceError": frozenset({"ComposerServiceError"}),
    "ComposerRuntimePreflightError": frozenset({"ComposerServiceError"}),
}
```

Append to `tests/unit/scripts/cicd/test_enforce_composer_catch_order.py` inside `TestComposerCatchOrderEnforcer`:

```python
    def test_runtime_preflight_error_inverted_order_fails(self, tmp_path: Path) -> None:
        """ComposerRuntimePreflightError must be caught before ComposerServiceError."""
        _make_routes_tree(
            tmp_path,
            "def f():\n"
            "    try:\n"
            "        pass\n"
            "    except ComposerServiceError as exc:\n"
            "        pass\n"
            "    except ComposerRuntimePreflightError as crash:\n"
            "        pass\n",
        )
        result = _run(["check", "--root", str(tmp_path)])
        assert result.returncode != 0
        assert "CCO1" in result.stdout
        assert "ComposerRuntimePreflightError" in result.stdout
```

Run:

```bash
uv run pytest -q tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
```

Expected: pass. The existing `TestHierarchyConsistency` tests should also pass; if they fail, the registry still does not match the real `ComposerServiceError` subclass tree. No change is required to `scripts/cicd/enforce_composer_exception_channel.py` for this subclass: that enforcer guards bare `TypeError`/`ValueError`/`UnicodeError` raises in composer tools and has no `ComposerServiceError` subclass registry.

- [ ] **Step 5: Add deterministic finalization in composer service**

In `src/elspeth/web/composer/service.py`, import `ComposerRuntimePreflightError` and remove any planned regex import.

Add helper:

```python
    @staticmethod
    def _runtime_preflight_failure_message(result: ValidationResult) -> str:
        if result.errors:
            first = result.errors[0]
            suggestion = f" Suggested fix: {first.suggestion}" if first.suggestion else ""
            return f"I cannot mark this pipeline complete yet because runtime preflight failed: {first.message}.{suggestion}"
        failed_checks = [check for check in result.checks if not check.passed]
        if failed_checks:
            return f"I cannot mark this pipeline complete yet because runtime preflight failed: {failed_checks[0].detail}."
        return "I cannot mark this pipeline complete yet because runtime preflight failed."

    async def _finalize_no_tool_response(
        self,
        *,
        content: str,
        state: CompositionState,
        initial_version: int,
        user_id: str | None,
        last_runtime_preflight: ValidationResult | None,
        runtime_preflight_cache: _RuntimePreflightCache,
        session_scope: str,
    ) -> ComposerResult:
        runtime_result = last_runtime_preflight
        if state.version > initial_version:
            runtime_result = await self._cached_runtime_preflight(
                state,
                user_id=user_id,
                cache=runtime_preflight_cache,
                initial_version=initial_version,
                session_scope=session_scope,
            )

        if runtime_result is None:
            return ComposerResult(message=content, state=state)

        if not runtime_result.is_valid:
            return ComposerResult(
                message=self._runtime_preflight_failure_message(runtime_result),
                state=state,
                runtime_preflight=runtime_result,
                raw_assistant_content=content,
            )

        return ComposerResult(message=content, state=state, runtime_preflight=runtime_result)
```

Unexpected preflight exceptions and timeouts are already converted by
`_cached_runtime_preflight()` into `ComposerRuntimePreflightError` with
partial-state preservation. Do not convert those exceptions into a user-fixable
`ValidationResult`.

- [ ] **Step 6: Replace no-tool return sites**

In both no-tool return sites inside `_compose_loop()`, return:

```python
                return await self._finalize_no_tool_response(
                    content=assistant_message.content or "",
                    state=state,
                    initial_version=initial_version,
                    user_id=user_id,
                    last_runtime_preflight=last_runtime_preflight,
                    runtime_preflight_cache=runtime_preflight_cache,
                    session_scope=session_scope,
                )
```

- [ ] **Step 7: Run final-gate and catch-order tests**

Run:

```bash
uv run pytest -q tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightFinalGate tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/composer/protocol.py src/elspeth/web/composer/service.py scripts/cicd/enforce_composer_catch_order.py tests/unit/web/composer/test_service.py tests/unit/scripts/cicd/test_enforce_composer_catch_order.py
git commit -m "fix(composer): gate final responses on runtime preflight"
```

---

### Task 5: Persist Runtime Validity And Raw Model Text Across All Composer Write Paths

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Read: `docs/guides/session-db-reset.md`
- Test: `tests/unit/web/sessions/test_routes.py`
- Test: `tests/unit/web/sessions/test_fork.py`

- [ ] **Step 0: Apply the session DB reset runbook before schema-change rollout**

This task adds `chat_messages.raw_content`. `initialize_session_schema()` intentionally validates exact table schemas and raises `SessionSchemaError` for stale DB files; there is no migration path in this project. Before running backend tests against persistent dev data, and before restarting `elspeth.foundryside.dev`, delete or archive existing session DB files so they are recreated with the new column.

Do not duplicate the systemd/restart/env-file checklist in this implementation plan. Use the operational runbook at `docs/guides/session-db-reset.md`.

Before any staging reset, run the runbook's Landscape reference gate:

```bash
rg -n "session_id|chat_message_id|composition_state_id" src/elspeth/core/landscape
```

Expected for the current architecture: no output. If this finds a Landscape reference, stop and analyze the relationship before deleting `sessions.db`; the reset must not orphan audit rows. After the gate passes, follow the runbook to resolve `WebSettings.get_session_db_url()`, archive/delete only the confirmed session DB, restart staging with the approved host-side mechanism, and verify `/api/health`.

- [ ] **Step 1: Add route/helper tests for state persistence semantics**

Append to `tests/unit/web/sessions/test_routes.py`:

```python
def test_runtime_preflight_errors_are_used_for_composition_state_persistence() -> None:
    from elspeth.web.sessions.routes import _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())
    runtime = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="plugin_instantiation",
                passed=False,
                detail="Invalid configuration for transform 'batch_stats'",
            )
        ],
        errors=[
            ValidationError(
                component_id="agg1",
                component_type="transform",
                message="Invalid configuration for transform 'batch_stats'",
                suggestion="Remove required_input_fields from batch-aware transform options.",
            )
        ],
    )

    is_valid, messages = _composer_persisted_validation(authoring, runtime)

    assert is_valid is False
    assert messages == ["Invalid configuration for transform 'batch_stats'"]


def test_authoring_validity_is_not_marked_valid_when_runtime_preflight_failed_internally() -> None:
    from elspeth.web.sessions.routes import _RUNTIME_PREFLIGHT_FAILED, _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())

    is_valid, messages = _composer_persisted_validation(authoring, _RUNTIME_PREFLIGHT_FAILED)

    assert is_valid is False
    assert messages == ["runtime_preflight_failed"]


def test_composer_persisted_validation_has_no_split_runtime_failed_knob() -> None:
    import inspect

    from elspeth.web.sessions.routes import _composer_persisted_validation

    params = inspect.signature(_composer_persisted_validation).parameters

    assert "preflight_failed" not in params
    assert list(params) == ["authoring", "runtime_preflight"]


def test_authoring_valid_state_without_runtime_outcome_is_rejected() -> None:
    from elspeth.web.sessions.routes import _composer_persisted_validation

    authoring = ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=())

    with pytest.raises(ValueError, match="requires runtime preflight outcome"):
        _composer_persisted_validation(authoring, None)


@pytest.mark.asyncio
async def test_state_data_from_composer_state_propagates_to_dict_errors() -> None:
    from elspeth.web.composer.state import ValidationEntry
    from elspeth.web.sessions.routes import _state_data_from_composer_state

    state = MagicMock(spec=CompositionState)
    state.version = 1
    state.validate.return_value = ValidationSummary(
        is_valid=False,
        errors=(ValidationEntry("validation", "validation_failed", "high"),),
        warnings=(),
        suggestions=(),
    )
    state.to_dict.side_effect = TypeError("broken Tier 1 state")

    with pytest.raises(TypeError, match="broken Tier 1 state"):
        await _state_data_from_composer_state(
            state,
            settings=object(),
            secret_service=None,
            user_id="user-1",
            runtime_preflight=None,
            preflight_exception_policy="persist_invalid",
            initial_version=None,
            log_prefix="compose",
            session_id=uuid.uuid4(),
        )


def test_runtime_preflight_telemetry_uses_bounded_attributes(monkeypatch) -> None:
    from elspeth.web.sessions import routes

    emitted: list[tuple[int, dict[str, str]]] = []

    class FakeCounter:
        def add(self, value: int, attributes: dict[str, str]) -> None:
            emitted.append((value, dict(attributes)))

    monkeypatch.setattr(routes, "_COMPOSER_RUNTIME_PREFLIGHT_COUNTER", FakeCounter())
    monkeypatch.setattr(routes, "_COMPOSER_AUTHORING_VALIDATION_COUNTER", FakeCounter())

    routes._record_composer_runtime_preflight_telemetry(
        "exception",
        source="compose",
        exception_class="RuntimeError",
    )
    routes._record_composer_runtime_preflight_telemetry(
        "exception",
        source="compose",
        exception_class="AdversarialPluginFailure_9c5dbf3e",
    )
    routes._record_composer_authoring_validation_telemetry(
        "exception",
        source="compose",
        exception_class="RuntimeError",
    )

    assert emitted == [
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "RuntimeError",
            },
        ),
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "other",
            },
        ),
        (
            1,
            {
                "result": "exception",
                "source": "compose",
                "exception_class": "RuntimeError",
            },
        ),
    ]
```

Add imports:

```python
from datetime import UTC, datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.web.composer.state import ValidationSummary
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationResult
from elspeth.web.sessions.protocol import ChatMessageRecord
```

Also append concrete write-path route tests so the split `is_valid` semantics from the prior review are mechanically pinned. These tests are the primary evidence that `composition_states.is_valid` reflects runtime preflight, not authoring-only validation:

```python
def _runtime_preflight_failed_result(message: str = "runtime preflight blocked export") -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="plugin_instantiation",
                passed=False,
                detail=message,
            )
        ],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message=message,
                suggestion=None,
            )
        ],
    )


def test_recompose_success_persists_runtime_invalid_state(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline"))
    finally:
        loop.close()

    changed_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="runtime-invalid-recompose"),
        version=_EMPTY_STATE.version + 1,
    )
    runtime_preflight = _runtime_preflight_failed_result("runtime failure from recompose")
    mock_composer = AsyncMock()
    mock_composer.compose = AsyncMock(
        return_value=ComposerResult(
            message="I cannot mark this pipeline complete yet.",
            state=changed_state,
            runtime_preflight=runtime_preflight,
        )
    )
    app.state.composer_service = mock_composer

    recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

    assert recompose_resp.status_code == 200
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "runtime-invalid-recompose"
    assert persisted.is_valid is False
    assert persisted.validation_errors == ["runtime failure from recompose"]


def test_recompose_convergence_persists_runtime_invalid_partial_state(tmp_path) -> None:
    from elspeth.web.composer.protocol import ComposerConvergenceError

    partial = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="partial-after-convergence"),
        version=2,
    )
    mock_composer = AsyncMock()
    mock_composer.compose = AsyncMock(
        side_effect=ComposerConvergenceError(
            max_turns=5,
            budget_exhausted="composition",
            partial_state=partial,
        )
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build a CSV pipeline"))
    finally:
        loop.close()

    runtime_preflight = _runtime_preflight_failed_result("runtime failure from convergence")
    with patch(
        "elspeth.web.sessions.routes._runtime_preflight_for_state",
        new=AsyncMock(return_value=runtime_preflight),
    ):
        recompose_resp = client.post(f"/api/sessions/{session_id}/recompose")

    assert recompose_resp.status_code == 422
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-convergence"
    assert persisted.is_valid is False
    assert persisted.validation_errors == ["runtime failure from convergence"]


def test_compose_plugin_crash_persists_runtime_invalid_partial_state(tmp_path) -> None:
    partial = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="partial-after-plugin-crash"),
        version=5,
    )
    original = ValueError("plugin bug after mutation")
    mock_composer = AsyncMock()
    mock_composer.compose = AsyncMock(
        side_effect=ComposerPluginCrashError(original, partial_state=partial),
    )

    app, service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/sessions", json={"title": "Test"})
    session_id = uuid.UUID(resp.json()["id"])

    runtime_preflight = _runtime_preflight_failed_result("runtime failure from plugin crash")
    with patch(
        "elspeth.web.sessions.routes._runtime_preflight_for_state",
        new=AsyncMock(return_value=runtime_preflight),
    ):
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Build me a pipeline"},
        )

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "composer_plugin_error"
    loop = asyncio.new_event_loop()
    try:
        persisted = loop.run_until_complete(service.get_current_state(session_id))
    finally:
        loop.close()
    assert persisted is not None
    assert persisted.metadata_ is not None
    assert persisted.metadata_["name"] == "partial-after-plugin-crash"
    assert persisted.is_valid is False
    assert persisted.validation_errors == ["runtime failure from plugin crash"]
```

The send-message success path is covered by the raw-content test below because it persists a changed state with an invalid `runtime_preflight`.

- [ ] **Step 2: Add raw content persistence test**

Append:

```python
def test_assistant_raw_content_is_persisted_but_not_returned(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)

    session_resp = client.post("/api/sessions", json={"title": "Chat"})
    session_id = uuid.UUID(session_resp.json()["id"])

    changed_state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="runtime preflight failed"),
        version=_EMPTY_STATE.version + 1,
    )
    composer_result = ComposerResult(
        message="I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
        state=changed_state,
        runtime_preflight=ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="bad config",
                    suggestion=None,
                )
            ],
        ),
        raw_assistant_content="The pipeline is complete and valid.",
    )
    composer = AsyncMock()
    composer.compose = AsyncMock(return_value=composer_result)
    app.state.composer_service = composer

    resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "build it"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"]["content"].startswith("I cannot mark this pipeline complete")
    assert "raw_content" not in body["message"]

    loop = asyncio.new_event_loop()
    try:
        messages = loop.run_until_complete(service.get_messages(session_id, limit=None))
    finally:
        loop.close()
    assistant = next(message for message in messages if message.role == "assistant")
    assert assistant.raw_content == "The pipeline is complete and valid."


def test_intercepted_assistant_history_is_annotated_without_raw_content() -> None:
    from elspeth.web.sessions.routes import (
        _INTERCEPTED_ASSISTANT_HISTORY_PREFIX,
        _composer_chat_history,
    )

    session_id = uuid.uuid4()
    message = ChatMessageRecord(
        id=uuid.uuid4(),
        session_id=session_id,
        role="assistant",
        content="I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
        raw_content="The pipeline is complete and valid.",
        tool_calls=None,
        created_at=datetime.now(UTC),
        composition_state_id=None,
    )

    history = _composer_chat_history([message])

    assert history == [
        {
            "role": "assistant",
            "content": (
                _INTERCEPTED_ASSISTANT_HISTORY_PREFIX
                + "I cannot mark this pipeline complete yet because runtime preflight failed: bad config."
            ),
        }
    ]
    assert "The pipeline is complete and valid" not in history[0]["content"]


def test_send_message_annotates_intercepted_assistant_history_for_llm(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    composer = _make_composer_mock(response_text="Retrying from the runtime failure.")
    app.state.composer_service = composer
    client = TestClient(app)

    session_resp = client.post("/api/sessions", json={"title": "Chat"})
    session_id = uuid.UUID(session_resp.json()["id"])

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(service.add_message(session_id, "user", "Build it"))
        loop.run_until_complete(
            service.add_message(
                session_id,
                "assistant",
                "I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
                raw_content="The pipeline is complete and valid.",
            )
        )
    finally:
        loop.close()

    resp = client.post(f"/api/sessions/{session_id}/messages", json={"content": "Fix it"})

    assert resp.status_code == 200
    history = composer.compose.call_args.args[1]
    assert history[1]["role"] == "assistant"
    assert history[1]["content"].startswith("[ELSPETH composer note: Your previous assistant response was intercepted")
    assert "runtime preflight failed: bad config" in history[1]["content"]
    assert "The pipeline is complete and valid" not in history[1]["content"]
```

Also append to `tests/unit/web/sessions/test_fork.py` inside `TestForkSession`:

```python
    @pytest.mark.asyncio
    async def test_fork_preserves_assistant_raw_content_for_copied_history(self, service) -> None:
        """Fork copies raw model provenance for historical assistant messages."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Build it")
        await service.add_message(
            session.id,
            "assistant",
            "I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
            raw_content="The pipeline is complete and valid.",
        )
        fork_msg = await service.add_message(session.id, "user", "Try again")

        _, messages, _ = await service.fork_session(
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Try a different way",
            user_id="alice",
            auth_provider_type="local",
        )

        copied_assistant = next(message for message in messages if message.role == "assistant")
        assert copied_assistant.content.startswith("I cannot mark this pipeline complete")
        assert copied_assistant.raw_content == "The pipeline is complete and valid."
        assert all(message.raw_content is None for message in messages if message.role in {"system", "user"})
```

Policy decision: forked sessions preserve `raw_content` for copied historical assistant messages. A fork represents a copied conversation history, and raw model text is part of that history's audit/provenance. Newly inserted system fork notices and edited user messages must have `raw_content is None`.

- [ ] **Step 3: Run failing route tests**

Run:

```bash
uv run pytest -q tests/unit/web/sessions/test_routes.py -k "runtime_preflight_errors_are_used or runtime_preflight_failed_internally or authoring_valid_state_without_runtime_outcome or state_data_from_composer_state_propagates_to_dict_errors or runtime_preflight_telemetry_uses_bounded_attributes or recompose_success_persists_runtime_invalid_state or recompose_convergence_persists_runtime_invalid_partial_state or compose_plugin_crash_persists_runtime_invalid_partial_state or raw_content or intercepted_assistant_history"
uv run pytest -q tests/unit/web/sessions/test_fork.py::TestForkSession::test_fork_preserves_assistant_raw_content_for_copied_history
```

Expected: fail because the helper, raw-content column, service parameter, fork propagation, and composer-history interception annotation do not exist yet.

- [ ] **Step 4: Add `chat_messages.raw_content` to session models and DTOs**

Modify `src/elspeth/web/sessions/models.py`:

```python
    Column("raw_content", Text, nullable=True),
```

Place it after `Column("content", Text, nullable=False)`.

Modify `ChatMessageRecord` in `src/elspeth/web/sessions/protocol.py`:

```python
    raw_content: str | None = None
```

Place it after `content` if test factories use keyword args. If positional construction breaks, update call sites to keywords.

Modify `SessionServiceProtocol.add_message()`:

```python
        raw_content: str | None = None,
```

Modify `SessionServiceImpl.add_message()` in `src/elspeth/web/sessions/service.py`:

```python
        raw_content: str | None = None,
```

Include it in the insert:

```python
                        raw_content=raw_content,
```

Include it in the returned `ChatMessageRecord`:

```python
            raw_content=raw_content,
```

Modify `_row_to_message_record()` or equivalent row conversion to pass `raw_content=row.raw_content`.

In `SessionServiceImpl.fork_session()`, update the explicit copied-message dict built from source messages to include:

```python
                    raw_content=msg.raw_content,
```

Leave the system fork notice and edited user message inserts with `raw_content=None`.

Do not add `raw_content` to `ChatMessageResponse` in `src/elspeth/web/sessions/schemas.py`; it is persisted for audit/provenance, not returned to normal clients.

Retention decision for this task: `raw_content` follows `chat_messages` retention, not Landscape payload retention. `elspeth purge --retention-days` does not touch chat session rows, and this PR must not silently wire raw model text into that purge path. Task 8 creates a separate low-priority tracker item to define chat/session `raw_content` lifecycle if the project wants retention beyond whole-session deletion.

- [ ] **Step 5: Add route validation helpers**

In `src/elspeth/web/sessions/routes.py`, import:

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

from opentelemetry import metrics

from elspeth.web.composer import yaml_generator
from elspeth.web.composer.protocol import ComposerRuntimePreflightError
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.execution.schemas import ValidationResult
```

If `routes.py` already imports `cast`, extend that import rather than adding a second `typing` import. Keep the existing `ValidationEntry` and `ChatMessageRecord` imports; the partial-state guard and composer-history helper need them.

Add helpers near `_state_response()`:

```python
_PreflightExceptionPolicy = Literal["raise", "persist_invalid"]
_ComposerPreflightTelemetryResult = Literal["passed", "failed", "exception"]
_ComposerPreflightTelemetrySource = Literal[
    "compose",
    "recompose",
    "compose_runtime_preflight",
    "recompose_runtime_preflight",
    "convergence",
    "plugin_crash",
    "yaml_export",
]


@dataclass(frozen=True, slots=True)
class _RuntimePreflightFailed:
    """Sentinel for internal preflight failure during composer state persistence."""

    exception_class: str | None = None


_RUNTIME_PREFLIGHT_FAILED = _RuntimePreflightFailed()
_RuntimePreflightOutcome = ValidationResult | _RuntimePreflightFailed | None

_COMPOSER_RUNTIME_PREFLIGHT_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.runtime_preflight.total",
    unit="1",
    description="Count of composer runtime preflight outcomes by route and result",
)
_COMPOSER_AUTHORING_VALIDATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.authoring_validation.total",
    unit="1",
    description="Count of composer authoring-state validation outcomes by route and result",
)

_INTERCEPTED_ASSISTANT_HISTORY_PREFIX = (
    "[ELSPETH composer note: Your previous assistant response was intercepted "
    "by runtime preflight and replaced before it was shown to the user. The "
    "visible replacement below is authoritative; continue from it and do not "
    "assume the original completion claim succeeded.]\n\n"
)

_COMPOSER_EXCEPTION_CLASS_BUCKETS = frozenset(
    {
        "AttributeError",
        "ComposerRuntimePreflightError",
        "FileNotFoundError",
        "GraphValidationError",
        "ImportError",
        "OSError",
        "PermissionError",
        "PluginConfigError",
        "PluginNotFoundError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)
_OTHER_COMPOSER_EXCEPTION_CLASS = "other"


def _bounded_composer_exception_class(exception_class: str | None) -> str | None:
    if exception_class is None:
        return None
    if exception_class in _COMPOSER_EXCEPTION_CLASS_BUCKETS:
        return exception_class
    return _OTHER_COMPOSER_EXCEPTION_CLASS


def _record_composer_runtime_preflight_telemetry(
    result: _ComposerPreflightTelemetryResult,
    *,
    source: _ComposerPreflightTelemetrySource,
    exception_class: str | None = None,
) -> None:
    attrs = {"result": result, "source": source}
    bounded_exception_class = _bounded_composer_exception_class(exception_class)
    if bounded_exception_class is not None:
        attrs["exception_class"] = bounded_exception_class
    _COMPOSER_RUNTIME_PREFLIGHT_COUNTER.add(1, attrs)


def _record_composer_authoring_validation_telemetry(
    result: _ComposerPreflightTelemetryResult,
    *,
    source: _ComposerPreflightTelemetrySource,
    exception_class: str | None = None,
) -> None:
    attrs = {"result": result, "source": source}
    bounded_exception_class = _bounded_composer_exception_class(exception_class)
    if bounded_exception_class is not None:
        attrs["exception_class"] = bounded_exception_class
    _COMPOSER_AUTHORING_VALIDATION_COUNTER.add(1, attrs)


def _composer_history_content(message: ChatMessageRecord) -> str:
    """Return the content sent back to the composer LLM for a stored message."""
    if message.role == "assistant" and message.raw_content is not None:
        return _INTERCEPTED_ASSISTANT_HISTORY_PREFIX + message.content
    return message.content


def _composer_chat_history(messages: Sequence[ChatMessageRecord]) -> list[dict[str, str]]:
    """Convert persisted session messages to LLM chat history.

    `raw_content` is attribution/audit data and must not be sent back to the
    model. When an assistant message has raw_content, its visible content is a
    synthetic runtime-preflight replacement; annotate that visible content so
    the next LLM turn understands why its apparent prior answer changed.
    """
    return [{"role": message.role, "content": _composer_history_content(message)} for message in messages]


def _composer_persisted_validation(
    authoring: ValidationSummary,
    runtime_preflight: _RuntimePreflightOutcome,
) -> tuple[bool, list[str] | None]:
    """Return persisted validity/errors for a composer-produced state."""
    if isinstance(runtime_preflight, _RuntimePreflightFailed):
        return False, ["runtime_preflight_failed"]
    if runtime_preflight is not None:
        messages = [error.message for error in runtime_preflight.errors]
        return runtime_preflight.is_valid, messages or None
    if authoring.is_valid:
        raise ValueError("Composer persistence for authoring-valid state requires runtime preflight outcome")
    messages = [error.message for error in authoring.errors]
    return authoring.is_valid, messages or None


async def _runtime_preflight_for_state(
    state: CompositionState,
    *,
    settings: Any,
    secret_service: Any | None,
    user_id: str | None,
) -> ValidationResult:
    return await asyncio.wait_for(
        run_sync_in_worker(
            validate_pipeline,
            state,
            settings,
            yaml_generator,
            secret_service=secret_service,
            user_id=user_id,
        ),
        timeout=settings.composer_runtime_preflight_timeout_seconds,
    )


async def _state_data_from_composer_state(
    state: CompositionState,
    *,
    settings: Any,
    secret_service: Any | None,
    user_id: str | None,
    runtime_preflight: _RuntimePreflightOutcome,
    preflight_exception_policy: _PreflightExceptionPolicy,
    initial_version: int | None,
    log_prefix: str,
    session_id: UUID,
) -> tuple[CompositionStateData, ValidationSummary]:
    # Session id remains available through persisted session state; keep
    # telemetry attributes bounded and do not tag metrics with per-session IDs.
    del session_id

    try:
        authoring = state.validate()
    except (ValueError, TypeError, KeyError) as val_err:
        _record_composer_authoring_validation_telemetry(
            "exception",
            source=cast(_ComposerPreflightTelemetrySource, log_prefix),
            exception_class=type(val_err).__name__,
        )
        authoring = ValidationSummary(
            is_valid=False,
            errors=(ValidationEntry("validation", "validation_failed", "high"),),
        )
    else:
        _record_composer_authoring_validation_telemetry(
            "passed" if authoring.is_valid else "failed",
            source=cast(_ComposerPreflightTelemetrySource, log_prefix),
        )

    runtime = runtime_preflight
    if runtime is None and authoring.is_valid:
        try:
            runtime = await _runtime_preflight_for_state(
                state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
            )
        except Exception as exc:
            if preflight_exception_policy == "raise":
                raise ComposerRuntimePreflightError.capture(
                    exc,
                    state=state,
                    initial_version=initial_version if initial_version is not None else state.version,
                ) from exc
            _record_composer_runtime_preflight_telemetry(
                "exception",
                source=cast(_ComposerPreflightTelemetrySource, log_prefix),
                exception_class=type(exc).__name__,
            )
            runtime = _RuntimePreflightFailed(type(exc).__name__)
    if isinstance(runtime, ValidationResult):
        _record_composer_runtime_preflight_telemetry(
            "passed" if runtime.is_valid else "failed",
            source=cast(_ComposerPreflightTelemetrySource, log_prefix),
        )
    persisted_is_valid, persisted_errors = _composer_persisted_validation(
        authoring,
        runtime,
    )
    state_d = state.to_dict()
    return (
        CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=persisted_is_valid,
            validation_errors=persisted_errors,
        ),
        authoring,
    )
```

This helper deliberately preserves the existing damaged-partial-state guard from `_handle_convergence_error()` and `_handle_plugin_crash()`: only `state.validate()` may catch `(ValueError, TypeError, KeyError)`. A `TypeError` or `KeyError` from `state.to_dict()` or `CompositionStateData(...)` is a Tier 1 invariant bug and must still propagate. Do not broaden the DB persistence guards around this helper; existing `SQLAlchemyError`-only catches remain the right boundary.

The `_RuntimePreflightOutcome` sentinel is intentional: callers cannot pass a successful `ValidationResult` and an independent "preflight failed" flag at the same time. If an implementer feels tempted to add another boolean to `_state_data_from_composer_state()`, stop and model it as a single outcome value instead.

Use `preflight_exception_policy="raise"` for normal `send_message` and `recompose` success paths so unexpected internal preflight failures become `ComposerRuntimePreflightError` and return the typed partial-state-preserving 500. Use `preflight_exception_policy="persist_invalid"` only inside `_handle_convergence_error()` and `_handle_plugin_crash()`, where preserving the original 422/500 response is more important than running preflight again. Do not add `slog` calls for these normal operational outcomes. Validation/preflight pass/fail/exception rates go to OpenTelemetry counters with bounded attributes (`result`, `source`, and sanitized `exception_class` only). Session correlation remains in persisted session rows and HTTP request context, not as a high-cardinality metric attribute.

- [ ] **Step 6: Update all four state write paths**

In `send_message`, replace:

```python
            chat_messages = [{"role": r.role, "content": r.content} for r in records[:-1]]
```

with:

```python
            chat_messages = _composer_chat_history(records[:-1])
```

In `recompose`, replace the same inline comprehension over `records[:-1]` with:

```python
            chat_messages = _composer_chat_history(records[:-1])
```

This is required because `composer/prompts.py::build_messages()` feeds the LLM
from `chat_history[*]["content"]` exactly as supplied by the route. Do not send
`raw_content` back to the model; the annotation plus visible replacement message
is the model-facing truth.

In `send_message` and `recompose`, replace local `result.state.validate()` / `CompositionStateData(...)` construction with:

```python
                state_data, validation = await _state_data_from_composer_state(
                    result.state,
                    settings=settings,
                    secret_service=request.app.state.scoped_secret_resolver,
                    user_id=str(user.user_id),
                    runtime_preflight=result.runtime_preflight,
                    preflight_exception_policy="raise",
                    initial_version=state.version,
                    log_prefix="compose",  # use "recompose" in the recompose handler
                    session_id=session.id,
                )
```

When saving the assistant message in both handlers, pass:

```python
                raw_content=result.raw_assistant_content,
```

Update `_handle_convergence_error()` and `_handle_plugin_crash()` signatures to accept:

```python
    settings: Any,
    secret_service: Any | None,
```

Inside both helpers, replace local `CompositionStateData(...)` construction with:

```python
            state_data, validation = await _state_data_from_composer_state(
                exc.partial_state,
                settings=settings,
                secret_service=secret_service,
                user_id=user_id,
                runtime_preflight=None,
                preflight_exception_policy="persist_invalid",
                initial_version=None,
                log_prefix=log_prefix,
                session_id=session_id,
            )
```

Update all call sites to pass `settings=request.app.state.settings` and `secret_service=request.app.state.scoped_secret_resolver`.

Add a catch for `ComposerRuntimePreflightError` before the generic `ComposerServiceError` catch in both `send_message` and `recompose`. Reuse `_handle_plugin_crash()` response semantics, but use `log_prefix` values `compose_runtime_preflight` and `recompose_runtime_preflight`, and raise HTTP 500 from `exc.original_exc`.

- [ ] **Step 7: Run route tests**

Run:

```bash
uv run pytest -q tests/unit/web/sessions/test_routes.py
uv run pytest -q tests/unit/web/sessions/test_fork.py::TestForkSession::test_fork_preserves_assistant_raw_content_for_copied_history
```

Expected: pass. Update the known in-file helper `_ProgressRouteSessionService.add_message()` to accept `raw_content: str | None = None` and pass it into `ChatMessageRecord`. `_make_composer_mock()` can continue omitting raw content because `ComposerResult.raw_assistant_content` defaults to `None`.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/sessions/models.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_fork.py
git commit -m "fix(web): persist runtime composer validity across write paths"
```

---

### Task 6: Gate YAML Export Against The Exact State Snapshot

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py`
- Test: `tests/unit/web/sessions/test_routes.py`

- [ ] **Step 1: Add failing exact-snapshot YAML export test**

Add near existing state YAML route tests:

If Task 5 has not already added it, update the test import to:

```python
from unittest.mock import AsyncMock, patch
```

```python
@pytest.mark.asyncio
async def test_get_state_yaml_validates_exact_state_snapshot(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    session = await service.create_session("alice", "Pipeline", "local")
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            source={
                "plugin": "csv",
                "on_success": "main",
                "options": {"path": "/data/blobs/input.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            outputs=[
                {
                    "name": "main",
                    "plugin": "csv",
                    "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
            metadata_={"name": "Snapshot", "description": ""},
            is_valid=True,
        ),
    )

    captured_states: list[CompositionState] = []

    async def fake_runtime_preflight(state, *, settings, secret_service, user_id):
        captured_states.append(state)
        return ValidationResult(is_valid=False, checks=[], errors=[
            ValidationError(
                component_id="source",
                component_type="source",
                message="runtime preflight failed for captured state",
                suggestion=None,
            )
        ])

    with patch("elspeth.web.sessions.routes._runtime_preflight_for_state", side_effect=fake_runtime_preflight):
        resp = client.get(f"/api/sessions/{session.id}/state/yaml")

    assert resp.status_code == 409
    assert "runtime preflight failed for captured state" in resp.json()["detail"]
    assert len(captured_states) == 1
    assert captured_states[0].source is not None


@pytest.mark.asyncio
async def test_get_state_yaml_does_not_export_resolved_secret_values(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    client = TestClient(app)
    resolved_secret = "__RESOLVED_SECRET_CANARY_DO_NOT_EXPORT__"

    class FakeResolvedSecretService:
        resolved_value = resolved_secret

    app.state.scoped_secret_resolver = FakeResolvedSecretService()
    session = await service.create_session("alice", "Pipeline", "local")
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            source={
                "plugin": "csv",
                "on_success": "main",
                "options": {
                    "path": "/data/blobs/input.csv",
                    "schema": {"mode": "observed"},
                    "api_key": {"secret_ref": "OPENAI_API_KEY"},
                },
                "on_validation_failure": "quarantine",
            },
            outputs=[
                {
                    "name": "main",
                    "plugin": "csv",
                    "options": {"path": "outputs/out.csv", "schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
            metadata_={"name": "Secret export", "description": ""},
            is_valid=True,
        ),
    )

    async def fake_runtime_preflight(state, *, settings, secret_service, user_id):
        assert secret_service is app.state.scoped_secret_resolver
        assert secret_service.resolved_value == resolved_secret
        # The runtime preflight path may resolve the secret in memory; export
        # must still serialize the original state snapshot with the secret_ref marker.
        assert state.to_dict()["source"]["options"]["api_key"] == {"secret_ref": "OPENAI_API_KEY"}
        return ValidationResult(is_valid=True, checks=[], errors=[])

    with patch("elspeth.web.sessions.routes._runtime_preflight_for_state", side_effect=fake_runtime_preflight):
        resp = client.get(f"/api/sessions/{session.id}/state/yaml")

    assert resp.status_code == 200
    exported_yaml = resp.json()["yaml"]
    assert resolved_secret not in exported_yaml
    parsed = yaml.safe_load(exported_yaml)
    assert parsed["source"]["options"]["api_key"] == {"secret_ref": "OPENAI_API_KEY"}
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run pytest -q tests/unit/web/sessions/test_routes.py::test_get_state_yaml_validates_exact_state_snapshot tests/unit/web/sessions/test_routes.py::test_get_state_yaml_does_not_export_resolved_secret_values
```

Expected: fail because `get_state_yaml()` currently validates authoring state only, and because no export regression pins the in-memory secret-resolution boundary yet.

- [ ] **Step 3: Update `get_state_yaml()`**

In `src/elspeth/web/sessions/routes.py`, replace authoring-only validation with:

```python
        state = _state_from_record(state_record)
        runtime_validation = await _runtime_preflight_for_state(
            state,
            settings=request.app.state.settings,
            secret_service=request.app.state.scoped_secret_resolver,
            user_id=str(user.user_id),
        )
        if not runtime_validation.is_valid:
            detail = "Current composition state failed runtime preflight. Fix validation errors before exporting YAML."
            if runtime_validation.errors:
                detail = f"{detail} First error: {runtime_validation.errors[0].message}"
            raise HTTPException(status_code=409, detail=detail)
        yaml_str = generate_yaml(state)
```

Do not call `execution_service.validate(session.id)` here; that method re-reads current state internally and can validate a different snapshot than the route serializes.

- [ ] **Step 4: Run YAML route tests**

Run:

```bash
uv run pytest -q tests/unit/web/sessions/test_routes.py -k state_yaml
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_routes.py
git commit -m "fix(web): validate exact state before yaml export"
```

---

### Task 7: Add Corrected Scenario-Level Composer Eval Regressions

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

- [ ] **Step 1: Add corrected preview/runtime scenario test**

Append:

```python
def test_runtime_preflight_preview_blocks_scenario_2_invalid_trigger(tmp_path: Path) -> None:
    """Scenario 2: preview must show runtime failure, not authoring-only validity."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    settings = _web_settings(data_dir)

    runtime_preflight = lambda candidate: validate_pipeline(candidate, settings, composer_yaml_generator)
    preview = execute_tool(
        "preview_pipeline",
        {},
        state,
        _mock_catalog(),
        data_dir=str(data_dir),
        runtime_preflight=runtime_preflight,
    )

    assert preview.success is True
    assert preview.data["is_valid"] is False
    assert preview.data["runtime_preflight"]["is_valid"] is False
    assert "end_of_source" in json.dumps(preview.data["runtime_preflight"])
```

- [ ] **Step 2: Add corrected final gate scenario test**

Append:

```python
@pytest.mark.asyncio
async def test_final_completion_claim_is_replaced_by_runtime_preflight_failure(tmp_path: Path) -> None:
    """The composer must not repeat an LLM complete/valid claim after dry-run failure."""
    data_dir, source_path, output_path = _scenario_2_files(tmp_path)
    settings = _web_settings(data_dir)
    composer = ComposerServiceImpl(catalog=_mock_catalog(), settings=settings)
    state = _aggregation_state(
        source_path,
        output_path,
        trigger={"condition": "end_of_source"},
        aggregation_options={"schema": {"mode": "observed"}, "value_field": "amount"},
    )
    changed_state = replace(state, version=state.version + 1)

    result = await composer._finalize_no_tool_response(
        content="The pipeline is complete and valid.",
        state=changed_state,
        initial_version=state.version,
        user_id=EVAL_USER_ID,
        last_runtime_preflight=None,
        runtime_preflight_cache=composer._new_runtime_preflight_cache(),
        session_scope="session:eval",
    )

    assert result.message != "The pipeline is complete and valid."
    assert result.raw_assistant_content == "The pipeline is complete and valid."
    assert result.runtime_preflight is not None
    assert result.runtime_preflight.is_valid is False
```

Add import if needed:

```python
from dataclasses import replace
```

- [ ] **Step 3: Run scenario tests**

Run:

```bash
uv run pytest -q tests/integration/pipeline/test_composer_llm_eval_characterization.py -k "runtime_preflight or final_completion_claim"
```

Expected: pass.

- [ ] **Step 4: Run full characterization suite**

Run:

```bash
uv run pytest -q tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

Expected: pass, respecting existing expected xfails.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(composer): cover runtime preflight eval scenarios"
```

---

### Task 8: Final Verification And Tracker Closeout

**Files:**
- No code files unless verification exposes a real regression.
- Update tracker issue `elspeth-34baf10c01`.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run pytest -q tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_preflight_side_effects.py tests/unit/web/execution/test_runtime_preflight_coordinator.py tests/unit/web/execution/test_service.py tests/unit/web/composer/test_tools.py::TestPreviewPipeline tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightCacheAndTimeout tests/unit/web/composer/test_service.py::TestComposerRuntimePreflightFinalGate tests/unit/composer_mcp/test_server.py::test_mcp_preview_runtime_preflight_joins_shared_session_inflight tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_fork.py::TestForkSession::test_fork_preserves_assistant_raw_content_for_copied_history tests/unit/scripts/cicd/test_enforce_composer_catch_order.py tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py -k "state_yaml or runtime_preflight or authoring_valid_state_without_runtime_outcome or state_data_from_composer_state_propagates_to_dict_errors or raw_content or intercepted_assistant_history or composer or catch_order or preflight_mode or mcp_preview"
```

Expected: pass.

- [ ] **Step 2: Run composer eval characterization**

Run:

```bash
uv run pytest -q tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

Expected: pass, respecting existing expected xfails.

- [ ] **Step 3: Run lint on touched files**

Run:

```bash
uv run ruff check src/elspeth/plugins/infrastructure/preflight.py src/elspeth/cli_helpers.py src/elspeth/web/app.py src/elspeth/web/config.py src/elspeth/web/execution/preflight.py src/elspeth/web/execution/runtime_preflight.py src/elspeth/web/execution/protocol.py src/elspeth/web/execution/validation.py src/elspeth/web/execution/service.py src/elspeth/web/composer/tools.py src/elspeth/web/composer/service.py src/elspeth/web/composer/protocol.py src/elspeth/composer_mcp/server.py src/elspeth/web/sessions/models.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py scripts/cicd/enforce_composer_catch_order.py tests/unit/web/execution/test_preflight_side_effects.py tests/unit/web/execution/test_runtime_preflight_coordinator.py tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_service.py tests/unit/composer_mcp/test_server.py tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_fork.py tests/unit/scripts/cicd/test_enforce_composer_catch_order.py tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

Expected: pass.

- [ ] **Step 4: Run type checking for touched packages**

Run:

```bash
uv run mypy src/elspeth/cli_helpers.py src/elspeth/plugins/infrastructure/preflight.py src/elspeth/web/app.py src/elspeth/web/config.py src/elspeth/web/execution src/elspeth/web/composer src/elspeth/web/sessions src/elspeth/composer_mcp
```

Expected: pass or existing unrelated baseline only. Fix new errors before continuing.

- [ ] **Step 5: Check dependent tracker context before closing**

Run:

```bash
filigree show elspeth-34baf10c01
filigree show elspeth-528bde62bb
filigree show elspeth-87f6d5dea5
filigree show elspeth-f936acd071
filigree show elspeth-dcf12c061b
```

Expected: confirm `elspeth-34baf10c01` itself is complete. Do not close sibling issues unless their acceptance criteria were explicitly implemented and verified.

- [ ] **Step 6: Track explicitly deferred follow-ups before closeout**

Search first to avoid duplicates:

```bash
filigree search "composer runtime validation Errors heading"
filigree search "raw_content retention chat messages"
```

If no matching issue exists for the frontend copy debt, create one:

```bash
filigree create "Composer validation copy — distinguish authoring and runtime errors" --type=task --priority=3 --description="Runtime-preflight failures may appear under the existing frontend Errors heading after elspeth-34baf10c01. Review SpecView.tsx/session validation copy and separate authoring validation from runtime preflight language if the combined heading is confusing."
```

If no matching issue exists for raw model text lifecycle, create one:

```bash
filigree create "Composer raw model retention — define raw_content lifecycle" --type=task --priority=3 --description="chat_messages.raw_content preserves replaced assistant model text for attribution. It follows chat message/session retention and is not covered by elspeth purge --retention-days. Decide whether whole-session deletion is sufficient or whether session/raw_content retention needs a dedicated purge policy."
```

- [ ] **Step 7: Add tracker evidence**

Run:

```bash
filigree add-comment elspeth-34baf10c01 "Implemented runtime-equivalent composer preflight. Verification: execution validation/service tests, composer preview/final-gate tests, session route persistence/YAML export tests, and composer LLM eval characterization suite."
```

- [ ] **Step 8: Close issue only if acceptance criteria are met**

Run:

```bash
filigree close elspeth-34baf10c01 --reason="Composer preview and final responses now use runtime preflight; persisted validity and YAML export are gated by runtime validation; targeted tests pass."
```

---

## Follow-Up Observations To Track Separately

- Runtime preflight currently resolves only `source.options.{path,file}` and `sinks.<name>.options.{path,file}`. If transform or aggregation plugins carry filesystem paths, create a separate issue for transform/aggregation path resolution.
- Runtime-flavored validation strings may appear under existing frontend "Errors" labels. Task 8 creates or confirms a low-priority tracker item for frontend copy; do not block runtime parity on it.
- `chat_messages.raw_content` is retained with chat messages and is not covered by `elspeth purge --retention-days`. Task 8 creates or confirms a low-priority tracker item for session/raw-content retention policy; do not add an unreviewed purge path in this PR.
- `validate_pipeline()` runs named validation steps directly while execution uses `build_validated_runtime_graph()`. Keep this split for now so preflight can report separate check names; if another runtime graph check is added, update the named `RUNTIME_CHECK_*` constants, `RUNTIME_GRAPH_VALIDATION_CHECKS`, and the Task 2 order/superset tests together.
- `execute_tool()` has a one-off `preview_pipeline` dispatch branch for `runtime_preflight`. Do not generalize it now. If a second composer tool needs runtime preflight, replace the special case with an explicit tool execution context or capability hook instead of adding more tool-name branches.
- `elspeth-dcf12c061b` remains relevant for LLM transform authoring. This preflight work can still land independently because runtime validation catches plugin config failures, but LLM schema guidance may still need its own fix.

## Self-Review

Spec coverage:

- Composer validation round-trips generated YAML through runtime settings loader: Tasks 1 and 2.
- Path allowlist and relative path parity: Task 1.
- Trigger grammar, batch dispatch, and plugin pre-execution blockers composer-visible: Tasks 3 and 7.
- Composer cannot emit complete-and-valid claims after runtime preflight failure: Task 4.
- Raw LLM content remains attributable when visible text is replaced: Tasks 4 and 5.
- The LLM's subsequent-turn history remains truthful when visible assistant text was replaced after runtime preflight: Task 5.
- Raw LLM content survives session fork history copying by explicit policy and regression test: Task 5.
- Raw LLM content retention is documented as chat-message/session lifecycle, with separate tracker follow-up for dedicated retention policy: Tasks 5 and 8.
- Persisted state truthfulness across all composer write paths: Task 5.
- Tier 1 propagation from `state.to_dict()` is tested instead of relying only on prose: Task 5.
- Existing session DB files are an operator-handled bootstrap boundary for the `raw_content` schema change: Task 5 Step 0 delegates operational details to `docs/guides/session-db-reset.md`, including the pre-rollout Landscape reference gate that prevents orphaning audit rows.
- YAML export truthfulness without a double-read race: Task 6.
- YAML export secret safety is negative-tested: resolved secret canaries stay out of exported YAML and `secret_ref` markers remain intact: Task 6.
- Runtime patch-target drift is enforced by persistent pytest coverage under `tests/unit/scripts/cicd/`, not by a one-shot grep: Task 2.
- Side-effect-free runtime preflight is enforced mechanically: `instantiate_plugins_from_config(..., preflight_mode=True)` sets an explicit plugin preflight context inside plugin construction, representative external constructors are covered by fake-socket tests, a separate worker-path regression proves constructors observe preflight mode through `run_sync_in_worker()` -> `validate_pipeline()`, and runtime execution keeps `preflight_mode=False`: Task 2.
- Runtime preflight latency is bounded and deduplicated: composer has a dedicated positive timeout setting, precomputes preview preflight outside the side-effectful tool worker, caches results/failures by `(session_scope, state.version, settings_hash)` for each compose call, and negative-tests timeout caching: Task 3.
- HTTP composer and composer MCP preview concurrency is covered: both surfaces use `RuntimePreflightCoordinator` for same-process in-flight singleflight keyed by logical session scope, and standalone cross-process MCP relies on side-effect-free `preflight_mode=True` rather than a distributed lock: Task 3.
- Scenario 1B/3 blob and Scenario 2 aggregation coverage: Task 7 plus existing characterization tests.
- Audit/logging policy: no row-level logs or normal-operational `slog` calls added. Pre-execution validation facts with probative value live in `composition_states`, chat message provenance, and HTTP responses; operational pass/fail/exception-rate visibility uses OpenTelemetry counters (`composer.runtime_preflight.total` and `composer.authoring_validation.total`) with bounded attributes only. `exception_class` is a closed bucket list plus `other`; never attach raw adversarial plugin class names to OTel attributes.
- Accepted low-priority architecture debt is visible: the two-arity runtime helper split and one-off `preview_pipeline` dispatch branch are deferred until a second caller makes an abstraction worthwhile.

Placeholder scan:

- No red-flag placeholder markers from the writing-plans checklist are present in implementation steps.

Type consistency:

- `ValidationSettings` is the protocol accepted by `validate_pipeline()`.
- `RuntimePreflight` is a callable from `CompositionState` to `ValidationResult`.
- `RuntimePreflightCoordinator` returns `ValidationResult | RuntimePreflightFailure`; callers translate failures into their own HTTP/MCP error surfaces.
- `RuntimePreflightKey` contains `session_scope`, `state_version`, and `settings_hash`; transport name is deliberately not part of the key.
- `composer_runtime_preflight_timeout_seconds` is a `WebSettings` / `ComposerSettings` field owned by the async composer boundary, not by synchronous `ValidationSettings`.
- `runtime_preflight_settings_hash()` includes only non-secret validation-affecting settings. Today that is resolved `data_dir`; future additions must be explicit and covered by the non-secret hash test.
- `RuntimeGraphBundle.plugin_bundle` is typed as `PluginBundle`; `RuntimeGraphBundle` is not frozen because it carries mutable `ExecutionGraph`.
- `ComposerResult.runtime_preflight` is `ValidationResult | None`.
- `ComposerResult.raw_assistant_content` is `str | None`.
- `ComposerRuntimePreflightError.original_exc` is `Exception`, matching the ordinary-exception preflight capture sites.
- `ToolResult.runtime_preflight` is `ValidationResult | None`.
- `ChatMessageRecord.raw_content` is persisted but not exposed in `ChatMessageResponse`.
- `_composer_chat_history()` returns `list[dict[str, str]]` for `ComposerService.compose()` and annotates intercepted assistant messages without including `raw_content`.
- `_state_data_from_composer_state()` takes one `_RuntimePreflightOutcome` value instead of split runtime-result and failure-boolean parameters.
- Task 4 depends on Task 3 because `ComposerServiceImpl._settings` is introduced there.
- Runtime graph validation labels use named `RUNTIME_CHECK_*` constants plus an equality assertion against `RUNTIME_GRAPH_VALIDATION_CHECKS`; do not reintroduce positional tuple unpacking.
- `tests/unit/scripts/cicd/test_runtime_preflight_patch_targets.py` owns stale execution-service patch-target detection.
