# Refactor: src/elspeth/web/composer/tools.py → composer/tools/ subpackage

**Branch:** `refactor/composer-tools-split` (off RC5.2)
**Worktree:** `.worktrees/refactor-composer-tools-split/`
**Baseline:** `tests/unit/web/composer/` — 2014 passed in 10.79s.

## Starting state

- `src/elspeth/web/composer/tools.py` — 7,389 lines, 23 classes, 134 top-level defs.
- `tests/unit/web/composer/test_tools.py` — 10,052 lines (out of scope for this refactor; will shrink as a downstream consequence).
- External call sites (production): 4 — `composer_mcp/server.py`, `composer/guided/steps.py`, `composer/service.py`, `sessions/routes/_helpers.py` (the last one imports `_DATA_ERROR_KEY` plus `execute_tool`).
- External call sites (tests): ~30 files; many private-symbol imports plus ~20 `patch("elspeth.web.composer.tools._X", …)` sites — these are the dotted-path patch traps.

## Target shape

```
src/elspeth/web/composer/tools/
  __init__.py        # Public re-export surface (back-compat for all current imports)
  _common.py         # ToolResult, _failure_result, _mutation_result, _discovery_result,
                     #   _attach_post_call_hints, _apply_merge_patch, _validate_mutation_arguments,
                     #   _vf_destination_note, diff_states, _prepend_rejection_entry,
                     #   _serialize_source/node/output/edge, _serialize_full_pipeline_state,
                     #   shared TypedDicts (_RepairToolCall, _AffectedConsumer, etc.)
  _registry.py       # _DISCOVERY_TOOLS, _MUTATION_TOOLS, _BLOB_DISCOVERY_TOOLS,
                     #   _BLOB_MUTATION_TOOLS, _CACHEABLE_DISCOVERY_TOOLS,
                     #   is_discovery_tool, is_mutation_tool, is_blob_store_only_mutation_tool,
                     #   is_cacheable_discovery_tool, is_session_aware_tool,
                     #   get_tool_definitions
  _dispatch.py       # execute_tool, _inject_prior_validation, _DATA_ERROR_KEY
  sessions.py        # set_pipeline, get_pipeline_state, request_interpretation_review,
                     #   set_metadata (pipeline metadata is session-scoped),
                     #   pipeline-state full-payload assembly
  sources.py         # set_source, clear_source, patch_source_options, inspect_source,
                     #   set_source_from_blob,
                     #   CSV inline-source helpers (_sample_csv_rows, _first_nonempty_csv_row,
                     #   _header_only_inline_csv_conflict, …)
  secrets.py         # list_secret_refs, validate_secret_ref, wire_secret_ref
                     #   (the wire handler reaches across source/node/output — its own concern)
  transforms.py      # upsert_node, remove_node, patch_node_options, upsert_edge, remove_edge,
                     #   list_transforms, node-routing helpers
  sinks.py           # list_sinks, _validate_sink_path, _prevalidate_sink (small file)
  outputs.py         # set_output, remove_output, patch_output_options
  recipes.py         # list_recipes, apply_pipeline_recipe
  generation.py      # preview_pipeline, diff_pipeline, explain_validation_error,
                     #   get_plugin_schema, get_plugin_assistance, get_audit_info,
                     #   list_models, get_expression_grammar,
                     #   compute_proof_diagnostics + diagnostic-code constants
  blobs.py           # The full blob subsystem (largest single plane, ~1500 lines):
                     #   _BLOB_QUOTA_BYTES, _ALLOWED_BLOB_MIME_TYPES, _SESSION_BLOB_LOCKS,
                     #   _check_blob_quota, _session_blob_lock, _sync_get_blob,
                     #   _execute_create_blob, _execute_update_blob, _execute_delete_blob,
                     #   _execute_get_blob_content, _prepare_blob_create, _persist_prepared_blob_create,
                     #   handle_list_blobs, handle_get_blob_metadata, _verify_blob_content_integrity
```

13 files: 7 operator-named planes (sessions/sources/transforms/sinks/outputs/recipes/generation)
plus 3 internal (`_common`, `_registry`, `_dispatch`) plus 1 necessary plane (`blobs`)
plus the package `__init__.py`.

## Single-registry pattern

Each plane module exposes:

```python
DISCOVERY_HANDLERS: dict[str, ToolHandler] = {...}
MUTATION_HANDLERS:  dict[str, ToolHandler] = {...}
```

`_registry.py` assembles them:

```python
from . import sources, transforms, sinks, outputs, recipes, generation, sessions, blobs
_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    **sources.DISCOVERY_HANDLERS, **transforms.DISCOVERY_HANDLERS,
    **sinks.DISCOVERY_HANDLERS, **outputs.DISCOVERY_HANDLERS,
    **recipes.DISCOVERY_HANDLERS, **generation.DISCOVERY_HANDLERS,
    **sessions.DISCOVERY_HANDLERS,
}
_MUTATION_TOOLS: dict[str, ToolHandler] = { ... same shape ... }
```

This is the "single registry" the operator named — assembled from per-plane contributions.

## Patch-compatibility strategy

The blast-radius concern: `patch("elspeth.web.composer.tools._BLOB_QUOTA_BYTES", 10)` patches the binding in the `tools` module namespace. After the split, the helpers in `blobs.py` read the constant via *their own* module namespace, so the test patch on the old path is a no-op.

**Resolution:** move the constant AND update the patch path in the same commit. Patches affected:

- `_BLOB_QUOTA_BYTES` — 6 patch sites → `elspeth.web.composer.tools.blobs._BLOB_QUOTA_BYTES`
- `_check_blob_quota` — 2 patch sites → `elspeth.web.composer.tools.blobs._check_blob_quota`
- `_sync_get_blob` — 1 patch site → `elspeth.web.composer.tools.blobs._sync_get_blob`

Other private-symbol imports (`_execute_create_blob`, `_session_blob_lock`, `_DISCOVERY_TOOLS`, …) get re-exported from `tools/__init__.py` so `from elspeth.web.composer.tools import …` keeps working unchanged.

## Sequencing — slice-then-verify

Each slice is one commit. After each, run the targeted test subset; full surface only at the end.

1. **Scaffold the package, preserve behaviour.** Replace `tools.py` with `tools/__init__.py` that contains every line of the old file. Verify: full composer test slice still passes.
2. **Extract `_common.py`** — pure helpers nobody patches; ToolResult + result/format helpers. Verify.
3. **Extract `blobs.py` + update test patch paths.** Biggest single plane, highest patch-density. Verify.
4. **Extract `sources.py`.** Verify.
5. **Extract `transforms.py`** (nodes + edges + metadata-as-pipeline-metadata moved to sessions). Verify.
6. **Extract `outputs.py`.** Verify.
7. **Extract `sinks.py`.** Verify.
8. **Extract `recipes.py`.** Verify.
9. **Extract `generation.py`** + update `_BLOCKING_DIAGNOSTIC_CODES` patch path. Verify.
10. **Extract `sessions.py`** (set_pipeline, get_pipeline_state, interpretation review). Verify.
11. **Extract `_registry.py` + `_dispatch.py`** — registries pulled out of `__init__.py`; `__init__.py` becomes the public facade only. Update `_DISCOVERY_TOOLS`/`_MUTATION_TOOLS` patch paths. Verify.
12. **Final pass:** full pytest + mypy + ruff + tier-model lint.

## Out of scope (this refactor)

- Splitting `tests/unit/web/composer/test_tools.py` (10k lines). It will shrink as a downstream consequence; not part of *this* PR.
- Renaming any public symbol. All current imports from `elspeth.web.composer.tools` must keep working.
- Behaviour changes. This is a structural refactor only.

## Bug-handling protocol (per operator directive 2026-05-23)

- Bugs found in `tools.py` / the new submodules — fix in this branch.
- Bugs found in other files — log in filigree, do not fix here.
