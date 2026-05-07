# PipelineConfig Immutable Type Annotations

**Date:** 2026-03-22
**Status:** Reviewed
**Scope:** Targeted refactoring — ~10 files, ~30 annotation changes + new test class + CI linter script
**Reviewed by:** Architecture critic, systems thinker, Python engineer, quality engineer

## Problem

`PipelineConfig` is a `frozen=True` dataclass whose `__post_init__` converts mutable container fields to immutable equivalents (`list` → `tuple`, `dict` → `MappingProxyType`). However, the field annotations still declare `list` and `dict`, creating a broken feedback loop:

- **mypy sees** `list[RowPlugin]` and allows `.append()`, `[i] = ...`
- **Runtime type is** `tuple[RowPlugin, ...]` which rejects those operations
- The type checker actively misleads developers instead of preventing mutation bugs

The real value of fixing this is not cosmetic annotation correctness — it's **closing the feedback loop** so mypy prevents mutation bugs at write time (seconds) instead of runtime (minutes to hours). Without this change, mypy gives silent permission to write code that will crash.

This was introduced in commit `c8205de3` which added runtime freezing to fix `elspeth-4ae10e14fb` but kept `list`/`dict` annotations to avoid cascading mypy errors at call sites. The annotation-vs-runtime mismatch is the remaining gap.

## Context

The rest of the codebase already follows the correct pattern:

- `PluginBundle` uses `Sequence[WiredTransform]`, `Mapping[str, SinkProtocol]`
- `GraphArtifacts` uses `Mapping[...]` for all its frozen dict fields
- `CoalesceMetadata` uses `MappingProxyType[str, str]` directly
- `accumulate_row_outcomes` already accepts `Mapping[str, object]`
- `_process_flush_results` already accepts `Mapping[str, object]`

`PipelineConfig` is the only frozen dataclass that still uses mutable annotations.

## Design

### PipelineConfig Field Annotations

Update 6 field annotations in `engine/orchestrator/types.py`:

| Field | Before | After |
|---|---|---|
| `transforms` | `list[RowPlugin]` | `Sequence[RowPlugin]` |
| `sinks` | `dict[str, SinkProtocol]` | `Mapping[str, SinkProtocol]` |
| `config` | `dict[str, Any]` | `Mapping[str, Any]` |
| `gates` | `list[GateSettings]` | `Sequence[GateSettings]` |
| `aggregation_settings` | `dict[str, AggregationSettings]` | `Mapping[str, AggregationSettings]` |
| `coalesce_settings` | `list[CoalesceSettings]` | `Sequence[CoalesceSettings]` |

The `Sequence` and `Mapping` imports are already present in `types.py`. Other changed files must be checked for `from collections.abc import Mapping, Sequence` — add where missing.

**Type system note:** `Sequence` is covariant, `Mapping` is covariant in the value type. Both trivially satisfied here — no invariance issues with mypy.

### Downstream Signature Updates

Every function that receives a PipelineConfig field (or a GraphArtifacts field) as a parameter and annotates it with a concrete `list`/`dict` type must be updated to use the abstract equivalent.

**`engine/orchestrator/core.py`:**

| Function | Param | Before | After |
|---|---|---|---|
| `_assign_plugin_node_ids` | `transforms` | `list[RowPlugin]` | `Sequence[RowPlugin]` |
| `_assign_plugin_node_ids` | `sinks` | `dict[str, SinkProtocol]` | `Mapping[str, SinkProtocol]` |
| `_assign_plugin_node_ids` | `transform_id_map` | `dict[int, NodeID]` | `Mapping[int, NodeID]` |
| `_assign_plugin_node_ids` | `sink_id_map` | `dict[SinkName, NodeID]` | `Mapping[SinkName, NodeID]` |

Note: `transform_id_map` and `sink_id_map` originate from `GraphArtifacts` which already uses `Mapping` annotations. `_flush_and_write_sinks` already uses `Mapping[SinkName, NodeID]` — no change needed there. These function signatures are the remaining gaps where the concrete type leaks through.

**`engine/orchestrator/validation.py`:**

| Function | Param | Before | After |
|---|---|---|---|
| `validate_route_destinations` | `route_resolution_map` | `dict[tuple[NodeID, str], RouteDestination]` | `Mapping[tuple[NodeID, str], RouteDestination]` |
| `validate_route_destinations` | `transform_id_map` | `dict[int, NodeID]` | `Mapping[int, NodeID]` |
| `validate_route_destinations` | `transforms` | `list[RowPlugin]` | `Sequence[RowPlugin]` |
| `validate_route_destinations` | `config_gate_id_map` | `dict[GateName, NodeID] \| None` | `Mapping[GateName, NodeID] \| None` |
| `validate_route_destinations` | `config_gates` | `list[GateSettings] \| None` | `Sequence[GateSettings] \| None` |
| `validate_transform_error_sinks` | `transforms` | `list[RowPlugin]` | `Sequence[RowPlugin]` |

**`engine/orchestrator/export.py`:**

| Function | Param | Before | After |
|---|---|---|---|
| `export_landscape` | `sinks` | `dict[str, SinkProtocol]` | `Mapping[str, SinkProtocol]` |

**`engine/processor.py`:**

| Function | Param | Before | After |
|---|---|---|---|
| `process_row` | `transforms` | `list[Any]` | `Sequence[RowPlugin]` |
| `process_existing_row` | `transforms` | `list[Any]` | `Sequence[RowPlugin]` |

Note: the existing `list[Any]` annotations are doubly wrong — both the container type and the element type defeat mypy. This change fixes both. If mypy errors cascade from the `RowPlugin` narrowing, fall back to `Sequence[Any]` and track the element type fix separately. Cascade risk is low — the `transforms` parameter is only used for a truthiness guard (`if transforms and ...`), never iterated or indexed by element.

`processor.py` imports `Mapping` but not `Sequence` — add `Sequence` to the `from collections.abc import` line.

**`contracts/plugin_context.py`:**

| Field | Before | After |
|---|---|---|
| `PluginContext.config` | `dict[str, Any]` | `Mapping[str, Any]` |

`PluginContext` is not frozen — it's a plain `@dataclass`. But its `config` field receives `PipelineConfig.config` which is `MappingProxyType` at runtime. Grep confirmed zero `ctx.config[...] = ...` mutation sites (the only `.config[...] =` writes are on `NodeInfo.config` in `builder.py` — a separate type). The annotation change makes the read-only contract explicit. `canonical_json` and `stable_hash` in the serialization path already dispatch on `isinstance(data, Mapping)`, so `MappingProxyType` values serialize correctly.

**`core/landscape/recorder.py`:**

| Function | Param | Before | After |
|---|---|---|---|
| `begin_run` | `config` | `dict[str, Any]` | `Mapping[str, Any]` |

**`core/landscape/run_lifecycle_repository.py`:**

`LandscapeRecorder.begin_run` delegates to `RunLifecycleRepository.begin_run` — both signatures need updating.

| Function | Param | Before | After |
|---|---|---|---|
| `begin_run` | `config` | `dict[str, Any]` | `Mapping[str, Any]` |

### Test Factory

**`testing/__init__.py`:** No changes needed. The `make_pipeline_config` factory accepts `list`/`dict` from callers and passes them to `PipelineConfig()` where `__post_init__` freezes them. Factory inputs should remain mutable types — `Sequence`/`Mapping` on input parameters would misleadingly suggest callers should pass immutable containers.

### New Tests

Add a `TestPipelineConfig` class to `tests/unit/engine/orchestrator/test_types.py`. This class currently does not exist, despite peer classes (`GraphArtifacts`, `AggregationFlushResult`) having equivalent tests in the same file.

Required tests:

1. **Freezing assertion** — construct with `list`/`dict` inputs, assert runtime types for all 6 fields:
   - `isinstance(config.transforms, tuple)`
   - `isinstance(config.sinks, MappingProxyType)`
   - `isinstance(config.config, MappingProxyType)`
   - `isinstance(config.gates, tuple)`
   - `isinstance(config.aggregation_settings, MappingProxyType)`
   - `isinstance(config.coalesce_settings, tuple)`

2. **Immutability assertion** — use `pytest.raises` (matching existing `TestAggregationFlushResult` pattern in same file):
   - `config.transforms` rejects `.append()` → `pytest.raises(AttributeError)`
   - `config.sinks` rejects `[key] = value` → `pytest.raises(TypeError)`
   - `config.config` rejects `[key] = value` → `pytest.raises(TypeError)`
   - At minimum one list-origin and one dict-origin field; all 6 preferred.

3. **Idempotency assertion** — construct with already-frozen inputs (`tuple`, `MappingProxyType`), assert:
   - Construction succeeds (no double-wrapping crash)
   - Values are equal to the originals: `config.transforms == original_tuple`, `config.sinks == original_proxy`
   - Runtime types remain correct: `isinstance(config.transforms, tuple)`

### What Doesn't Change

- **`__post_init__` freezing**: stays as unconditional `tuple()`/`MappingProxyType(dict())` conversion
- **Construction sites** (cli.py, test factories): keep passing `list`/`dict` literals — `Sequence`/`Mapping` accepts both mutable and immutable forms
- **Functions already using abstract types**: `accumulate_row_outcomes(config_sinks: Mapping)`, `_process_flush_results(sinks: Mapping)` — no change needed
- **Runtime behavior**: zero changes — this is purely a type annotation refactoring

## Verification

1. **mypy clean**: `mypy src/` must pass with zero new errors. This is the primary success criterion — the whole point is making mypy enforce the immutability contract.
2. **All unit tests pass**: including the new `TestPipelineConfig` class.
3. **ruff + tier model enforcer pass**: no new lint or tier violations expected.

## Risks

**Low risk.** Every change is a type annotation widening (`list` → `Sequence`, `dict` → `Mapping`). Widening parameter types is always backwards-compatible — any code that passed a `list` before can still pass a `list` to a `Sequence` parameter. Grep confirmed zero mutation sites across all affected fields.

## CI Linter: Frozen Dataclass Annotation Guard

### Purpose

Prevent the pattern that created this gap from recurring. A CI script that detects mutable container annotations (`list[`, `dict[`, `set[`) on fields of `frozen=True` dataclasses. Without this, a future developer writing a new frozen dataclass with `list` annotations and `__post_init__` freezing would pass mypy, pass tests, and silently reintroduce the same gap.

### Approach

An AST-walking script at `scripts/cicd/enforce_frozen_annotations.py`, following the `enforce_tier_model.py` pattern:

- Walk all Python files under `src/elspeth/`
- Find `@dataclass(frozen=True)` classes (must also match `frozen=True, slots=True` and other keyword combinations)
- For each field, check if the annotation uses `list[`, `dict[`, or `set[`
- Report violations with file, line, class, and field name
- Support an allowlist YAML file (at `config/cicd/enforce_frozen_annotations/`) for legitimate exceptions

**Critical implementation detail:** 126 files in `src/elspeth/` use `from __future__ import annotations` (including `types.py` itself — the file that created this problem). Under PEP 563, all annotations become `ast.Constant` string values, not `ast.Subscript` nodes. The linter must use `ast.unparse()` on annotation nodes (which returns the source text for both AST forms) and check the unparsed string for `"list["`, `"dict["`, `"set["`. This handles both stringified and non-stringified annotations uniformly, and correctly catches union types like `list[X] | None`.

### Detection rules

| Annotation pattern | Violation | Suggested fix |
|---|---|---|
| `list[X]` on frozen field | Mutable container on immutable dataclass | `Sequence[X]` or `tuple[X, ...]` |
| `dict[K, V]` on frozen field | Mutable container on immutable dataclass | `Mapping[K, V]` or `MappingProxyType[K, V]` |
| `set[X]` on frozen field | Mutable container on immutable dataclass | `frozenset[X]` |

### CI integration

Add a pre-commit hook entry in `pyproject.toml` alongside `enforce-tier-model`, using the same pattern:

```toml
[tool.pytest.ini_options]  # (adjacent section for reference)
# New hook in .pre-commit-config.yaml:
# - id: enforce-frozen-annotations
#   name: Enforce Frozen Annotations
#   entry: .venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
```

### Linter Tests

Add tests at `tests/unit/cicd/test_enforce_frozen_annotations.py` (following the `test_enforce_tier_model.py` pattern). Required:

1. Frozen dataclass with `list[X]` field → violation reported
2. Frozen dataclass with `Sequence[X]` field → clean
3. Non-frozen dataclass with `list[X]` field → clean (not in scope)
4. Frozen dataclass with `list[X] | None` union → violation reported
5. File using `from __future__ import annotations` with `list[X]` → violation reported
6. Frozen dataclass with `frozen=True, slots=True` → detected correctly

### Scope

The script only checks `frozen=True` dataclasses. Non-frozen dataclasses with mutable container fields are intentional and not flagged. Fields with `Sequence`, `Mapping`, `tuple`, `frozenset`, `MappingProxyType` annotations pass cleanly.

## Future Work

**`PluginContext.config` read-only property** — enforce immutability via `@property` with no setter. Tracked as `elspeth-560ba0fa3d`. Separate scope due to high fanout (L0 contracts, `__slots__` complications).

## Ordering

Two commits:

1. **Annotation refactoring + tests** — PipelineConfig field annotations, downstream signatures, `TestPipelineConfig` class. Single atomic commit (mypy requires all changes together).
2. **CI linter** — `enforce_frozen_annotations.py` script + allowlist + pre-commit hook. Separate commit (independent tooling, no mypy dependency).
