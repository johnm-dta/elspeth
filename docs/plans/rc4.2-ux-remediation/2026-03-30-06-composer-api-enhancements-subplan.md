# RC4.2 UX Remediation — Composer API Enhancements Subplan

Date: 2026-03-30 (expanded 2026-03-31, corrected 2026-03-31, implemented 2026-03-31)
Status: Implemented
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This subplan groups the assistant/composer authoring API improvements that make
iterative pipeline editing cheaper and more expressive.

Included requirements:

- `REQ-API-03` — patch operations (patch_source_options, patch_node_options,
  patch_output_options)
- `REQ-API-04` — atomic full pipeline replacement (set_pipeline)

Primary surfaces:

- `CompositionState` mutation methods
- composer tool definitions and dispatch registry
- `ToolResult` serialization (unchanged — reuses existing shape)

Not included (moved to Wave 4 sub-plan 08):

- `REQ-API-05` (clear_source), `REQ-API-08` (explain validation errors),
  `REQ-API-09` (list models) — these are P3 and have no dependencies within
  Wave 3.

---

## 2. Goals

- Reduce token waste when the LLM needs to change one field in a source/node/
  sink config — patch instead of full rewrite.
- Support atomic first-pass pipeline creation in a single tool call, avoiding
  invalid intermediate states from sequential set_source + upsert_node +
  upsert_edge + set_output.
- Maintain existing validation-after-mutation contract: every mutation tool
  returns `ToolResult` with a `ValidationSummary`.

Non-goals:

- No changes to discovery tools.
- No changes to the composition budget model.
- No frontend changes — these are assistant-facing tools only.

---

## 3. Architecture Decisions

### AD-1: Shallow merge-patch for all patch operations

Patch tools apply shallow merge-patch semantics at the `options` dict level:

- Keys present in the patch overwrite the corresponding keys in the target.
- Keys set to `null` in the patch delete the corresponding key from the target.
- Keys absent from the patch are left unchanged.

This is a **shallow merge** on `options` only. To change a nested key inside
options, the caller must supply the full nested structure for that key. This
matches the requirement spec and is simple to implement and explain.

Implementation: `dict(target)` + iterate patch items with explicit `None`-key
deletion via `result.pop(key, None)`. No library dependency needed.

**Implementation note:** The helper docstring uses "Shallow merge-patch"
rather than citing RFC 7396 directly, because RFC 7396 specifies recursive
merge and our implementation is deliberately shallow (one level only). The
tool descriptions accurately describe the behaviour to the LLM.

### AD-2: Patch tools reuse existing state mutation methods

Patch tools do NOT require new methods on `CompositionState`. The flow is:

1. Read the current spec from state (e.g. `state.source`)
2. Apply the merge-patch to `spec.options`
3. Construct a new spec with the patched options
4. Call the existing state method (`state.with_source()`, `state.with_node()`,
   `state.with_output()`)

This keeps the state dataclass simple and avoids duplicating mutation logic.

### AD-3: `set_pipeline` constructs a fresh state, not a delta

`set_pipeline` is a full replacement, not a merge. The tool receives the
complete pipeline structure (source, nodes, edges, outputs, metadata) and
constructs a new `CompositionState` from scratch using the existing spec
constructors.

It does NOT use `CompositionState.from_dict()` directly because `from_dict()`
expects the internal serialization format (with version numbers etc.). Instead,
the tool:

1. Constructs `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`,
   `PipelineMetadata` from the tool arguments
2. Creates a new `CompositionState` with `version = current_version + 1`
3. Runs `validate()` on the result
4. Returns the standard `ToolResult`

This reuses the same validation and construction logic as the individual tools,
applied atomically.

### AD-4: `set_pipeline` validates plugin names via catalog

Each source/node/sink plugin name in the `set_pipeline` payload is validated
against the catalog, using the same shared `_validate_plugin_name` helper as
`set_source`, `upsert_node`, and `set_output`. This prevents the LLM from
inventing plugin names in an atomic call that would bypass the per-tool
validation.

### AD-5: `set_pipeline` counts as a single composition budget charge

Even though it replaces the entire state, `set_pipeline` is a single mutation
tool call and charges one composition turn. This is the primary benefit: the
LLM can build a full pipeline in one turn instead of N sequential mutations.

### AD-6: Source path allowlist enforced on all source-mutating tools

Any tool that can set or modify source `options` must call
`_validate_source_path(options, data_dir)` to enforce the S2 composer-time
path allowlist. This applies to: `set_source`, `set_pipeline`,
`patch_source_options`, and `set_source_from_blob`.

**Discovered during implementation:** The initial `set_pipeline` and
`patch_source_options` implementations omitted this check, creating a
security control bypass. Both were fixed during code review. The root cause
is that the allowlist check lives in the handler layer, not in
`CompositionState.with_source()`. A future refactor could move it into the
state model to make bypasses impossible, but that requires threading
`data_dir` into the state model — a larger change deferred for now.

### AD-7: LLM tool arguments are Tier 3 (untrusted) input

Tool arguments arrive from the LLM and may contain `null` or wrong types
where the JSON Schema declares `"type": "object"`. Patch handlers validate
that `patch` is a `dict` before calling `_apply_merge_patch`. The
`set_pipeline` handler uses `args.get("metadata") or {}` to handle both
absent and `null` metadata, and catches `AttributeError` alongside
`KeyError`/`TypeError` in spec construction.

---

## 4. Detailed Changes

### Phase 6A: Patch Operations (3 new tools)

#### Handler signature convention

All handlers registered in `_MUTATION_TOOLS` conform to the `ToolHandler`
type alias:

```python
ToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogServiceProtocol, str | None],
    ToolResult,
]
```

Parameter order: `(arguments, state, catalog, data_dir=None)`.

The codebase uses a two-layer pattern: a thin wrapper (e.g.
`_handle_patch_source_options`) satisfies the `ToolHandler` signature and
delegates to an inner function (e.g. `_execute_patch_source_options`) with a
tighter signature. The `patch_source_options` wrapper passes `data_dir`
through for path validation (AD-6); the node and output wrappers do not need
it.

#### 6A.4: Shared merge-patch helper

```python
def _apply_merge_patch(
    target: Mapping[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Shallow merge-patch: overwrite or delete top-level keys in target."""
    result = dict(target)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result
```

- `target` typed `Mapping[str, Any]` for frozen `MappingProxyType` compat
- `dict(target)` produces a mutable copy; returned dict is re-frozen by spec
  constructor's `freeze_fields()`
- `result.pop(key, None)` is not a tier model violation — operates on a local
  mutable dict, not a dataclass field

#### 6A.1: `patch_source_options`

Inner handler receives `data_dir` and calls `_validate_source_path` on the
patched options (AD-6). Validates `patch` is a `dict` before use (AD-7).
Uses `args["patch"]` (required parameter — direct access, not `.get()`).

#### 6A.2: `patch_node_options`

Looks up node by `node_id`. Validates `patch` is a `dict`. Constructs new
`NodeSpec` with all 13 fields preserved from `current`, only `options`
replaced. Uses `state.with_node()`.

#### 6A.3: `patch_output_options`

Looks up output by `sink_name`. Validates `patch` is a `dict`. Constructs
new `OutputSpec` with `options` replaced. Uses `state.with_output()`.

#### 6A.5: Register in dispatch

All three wrappers added to `_MUTATION_TOOLS`.

---

### Phase 6B: `set_pipeline` Tool

#### 6B.4: Extract shared plugin validation

Extracted `_validate_plugin_name(catalog, plugin_type, name) -> str | None`
from the inline try/except blocks in `_execute_set_source`,
`_execute_upsert_node`, and `_execute_set_output`. Refactored all three
handlers to use it. The `plugin_type` parameter is typed with `PluginKind`
(`Literal["source", "transform", "sink"]`).

#### 6B.1–6B.3: Tool definition, handler, and registration

`_execute_set_pipeline(args, state, catalog, data_dir)`:

1. Validates source plugin via `_validate_plugin_name`
2. Validates source path via `_validate_source_path` (AD-6)
3. Validates each node's plugin (only `transform`/`aggregation` with non-None
   plugin)
4. Validates each output's plugin
5. Constructs all specs with correct defaults matching individual handlers:
   - `SourceSpec.on_validation_failure` → `"quarantine"`
   - `NodeSpec.on_success`/`on_error` → `None`; `options` → `{}`
   - `OutputSpec.on_write_failure` → `"discard"`; `options` → `{}`
   - `PipelineMetadata.name` → `"Untitled Pipeline"` (matches class default)
   - `fork_to`/`branches` converted to `tuple` when present
6. Catches `(KeyError, TypeError, AttributeError)` from spec construction
   (AD-7)
7. Builds `CompositionState` with `version = state.version + 1`
8. Returns `_mutation_result` with all nodes + source + outputs as affected

Wrapper `_handle_set_pipeline` passes `data_dir` through.

---

### Phase 6C: Tests

22 new tests in `tests/unit/web/composer/test_tools.py`:

- **TestMergePatch** (6): overwrites, adds, deletes-null,
  preserves-unmentioned, empty-patch, MappingProxyType immutability
- **TestPatchSourceOptions** (4): updates-key, adds-key, deletes-key-null,
  no-source-fails
- **TestPatchNodeOptions** (2): updates-key (other fields preserved),
  unknown-node-fails
- **TestPatchOutputOptions** (2): updates-key, unknown-sink-fails
- **TestSetPipeline** (8): creates-valid-state, unknown-source-fails,
  unknown-node-fails (selective side_effect), unknown-sink-fails (selective
  side_effect), missing-required-field-fails, replaces-entire-state,
  version-increments, validation-runs

2 existing tests updated: tool count 19→23, mutation tool count 8→12.

All error-path tests verify `result.data["error"]` content. Test fixture
`_valid_pipeline_args()` uses proper channel names (`source_out`) for node
inputs.

---

## 5. Implementation Order (as executed)

```
Phase 6A (patch tools) ─── sequential ──────────────────┐
  6A.4  Shared merge-patch helper                        │
  6A.1–3  3 patch tools + handlers + wrappers            │
  6A.5  Register in dispatch                             │
                                                         │
Phase 6B (set_pipeline) ── after 6A (same file) ───────┤
  6B.4  Extract _validate_plugin_name, refactor 3 handlers│
  6B.1–3  set_pipeline tool + handler + registration     │
                                                         │
Phase 6C (tests) ── after 6A + 6B ──────────────────────┘
  22 unit tests across 5 test classes
```

6A and 6B were executed sequentially (not in parallel) because both modify
`tools.py`. Sub-plan 05 landed first.

---

## 6. Files Affected

### Modified

| File | Changes |
|------|---------|
| `src/elspeth/web/composer/tools.py` | 4 new tool definitions, 4 wrapper handlers, 4 inner handlers, `_apply_merge_patch` helper, `_validate_plugin_name` helper, refactor of 3 existing handlers, registry additions (tool count 19→23, mutation 8→12) |
| `tests/unit/web/composer/test_tools.py` | 22 new tests in 5 classes, 2 updated count assertions, `deep_thaw` moved to module-level import |

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM uses patch tools incorrectly (wrong key names, wrong nesting level) | Low | Tool descriptions are explicit about shallow merge. Validation runs after every mutation. |
| `set_pipeline` with invalid plugin names | Low | Validate each component type separately and return on the first failure. |
| Plugin validation extraction changes existing behaviour | Low | Mechanical refactor. Existing tests all pass. |
| Merge-patch on frozen `MappingProxyType` options | None | `_apply_merge_patch` accepts `Mapping`, returns `dict`. Spec constructor re-freezes. |
| **Source path allowlist bypass via new tools** | **Caught** | `set_pipeline` and `patch_source_options` initially omitted `_validate_source_path`. Both fixed during code review (AD-6). |
| **`null` or non-object LLM arguments** | **Caught** | Patch handlers validate `isinstance(patch, dict)`. `set_pipeline` uses `or {}` for metadata and catches `AttributeError` (AD-7). |
| Tier model enforcer new violations | Low | `isinstance` checks in patch handlers are Tier 3 boundary validation (allowlisted). |

---

## 8. Acceptance Criteria

- [x] `patch_source_options`, `patch_node_options`, `patch_output_options`
      tools exist and appear in LLM tool definitions.
- [x] Patch tools apply shallow merge-patch: overwrite, add, delete-on-null,
      preserve-absent.
- [x] Patch tools fail gracefully when target doesn't exist (no source,
      unknown node, unknown output).
- [x] Patch tools validate `patch` argument is a dict (AD-7).
- [x] `patch_source_options` enforces source path allowlist (AD-6).
- [x] `set_pipeline` tool exists and creates a complete state in one call.
- [x] `set_pipeline` validates all plugin names against the catalog.
- [x] `set_pipeline` enforces source path allowlist (AD-6).
- [x] `set_pipeline` handles `null` metadata gracefully (AD-7).
- [x] `set_pipeline` increments version by 1.
- [x] `set_pipeline` charges one composition budget turn.
- [x] `PipelineMetadata.name` defaults to `"Untitled Pipeline"` (matches
      class default).
- [x] All new tools return `ToolResult` with `ValidationSummary` including
      warnings and suggestions (from sub-plan 05).
- [x] All handler wrappers conform to `ToolHandler` signature.
- [x] Shared `_validate_plugin_name` helper used by `set_source`,
      `upsert_node`, `set_output`, and `set_pipeline`.
- [x] All 80 tool tests pass (22 new + 58 existing).
- [x] All error-path tests verify error message content.
- [x] Tier model enforcer passes.
- [x] mypy + ruff clean.
