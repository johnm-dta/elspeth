# Composer tools/ subpackage map — 2026-05-23

Worktree: `/home/john/elspeth/.worktrees/composer-tools-rearchitect/`
Subpackage root: `src/elspeth/web/composer/tools/`
Total: 12 files, 8 168 lines.

Conventions in this map:
- "External callers" = imports from outside `tools/`, verified by `grep -rn "from elspeth.web.composer.tools" src/` and `from .tools` under `src/elspeth/web/composer/`.
- "Sibling import" = `from elspeth.web.composer.tools.<X> import ...` inside this subpackage.
- "Cross-plane private coupling" = import of an `_underscore` symbol across plane boundaries (either from another `tools/*` file, from `web.composer.state`, or from `web.blobs.service`).
- All line citations are paths relative to repo root in the worktree.

---

## `__init__.py` (456 lines)

**Surface.** This file is the public facade. It re-exports 200+ symbols (every name in every leaf, including underscore-prefixed ones) and declares an explicit `__all__` list at lines 251–456 so mypy treats every re-export as explicit (per the file docstring at lines 19–22). The five external import sites are:

- `src/elspeth/web/composer/service.py:101` — imports both the predicates (`is_discovery_tool`, `is_mutation_tool`, `is_cacheable_discovery_tool`, `is_session_aware_tool`, `is_blob_store_only_mutation_tool`) and the full handler payload set; 19 distinct call-sites across `service.py` lines 549–3309.
- `src/elspeth/web/composer/guided/steps.py:33` — guided-mode wiring.
- `src/elspeth/web/composer/_required_paths_validator.py:29` — imports `get_tool_definitions` only.
- `src/elspeth/composer_mcp/server.py:43` — the MCP server bridge.
- `src/elspeth/web/composer/progress.py:18` — imports `is_discovery_tool` only.
- `src/elspeth/web/sessions/routes/_helpers.py:126` — imports `_DATA_ERROR_KEY, execute_tool`.

**Internal layering.** Pure re-export module. It imports from every other file in the subpackage, in order: `_common`, `_dispatch`, `blobs`, `generation`, `outputs`, `recipes`, `secrets`, `sessions`, `sinks`, `sources`, `transforms`.

**Cross-plane private-symbol coupling.** None originates here; this file only re-exports. However, by exposing every `_underscore` symbol through `__all__` (e.g. `_DATA_ERROR_KEY`, `_BLOB_QUOTA_BYTES`, `_SESSION_BLOB_LOCKS`, `_SESSION_AWARE_TOOL_HANDLERS`, `_validate_plugin_name`), it normalises the pattern of leaf modules consuming each other's privates via the facade — and makes the audit-ticket cross-plane couplings (C1, C4, C5, C6) discoverable but not enforced.

---

## `_dispatch.py` (1451 lines)

**Surface.** Exports for external use: `execute_tool`, `get_tool_definitions`, `is_discovery_tool`, `is_mutation_tool`, `is_cacheable_discovery_tool`, plus six registry constants (`_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, `_BLOB_DISCOVERY_TOOLS`, `_BLOB_MUTATION_TOOLS`, `_SECRET_DISCOVERY_TOOLS`, `_SECRET_MUTATION_TOOLS`, `_CACHEABLE_DISCOVERY_TOOLS`) and two debug aggregates (`_all_tools`, `_all_tools_v2`). The only direct external import is via the facade; `service.py` reaches the predicates and `execute_tool` through `tools/__init__.py`.

**Internal layering.** Imports from every leaf plane except `outputs` and `transforms` for *handlers* (outputs/transforms handlers come in but only for the `_MUTATION_TOOLS` registry wiring), and from `_common` for `ToolHandler`/`ToolResult`/`RuntimePreflight`/`_failure_result`. Sibling-import block at lines 17–87. Tool-definitions list runs lines 90–1133 (~1043 lines of JSON-Schema blobs).

**Cross-plane private-symbol coupling.**
- `from elspeth.web.composer.tools.blobs import _BLOB_PROVENANCE_MUTATION_TOOLS, _BLOB_QUOTA_MUTATION_TOOLS` (lines 26–27) — consumed at dispatch lines 1363 and 1370 to decide which blob handlers receive `max_blob_storage_per_session_bytes` and `user_message_id` kwargs.
- `from elspeth.web.composer.tools.sessions import _SESSION_AWARE_TOOL_HANDLERS` (line 62) — used only by the `_all_tools_v2` overlap-check assertion and the async/sync iscoroutinefunction sweep at lines 1410–1429.
- `from elspeth.web.composer.tools._common import _DEFAULT_SOURCE_VALIDATION_FAILURE, _SOURCE_VALIDATION_FAILURE_DESCRIPTION, _failure_result` (lines 18–24).

**Dispatcher shape.**
- **Hard-coded `if tool_name == ...` branches** that bypass the registry dispatch:
  - `if tool_name == "preview_pipeline":` at line 1301 — needs the extra `runtime_preflight`, `session_engine`, `session_id` kwargs.
  - `if tool_name == "diff_pipeline":` at line 1313 — needs `baseline` and `current_validation` kwargs.
  - `if tool_name == "set_pipeline":` at line 1326 — needs `session_engine`, `session_id`, `user_message_id`, `max_blob_storage_per_session_bytes`.
  These three tools are listed once in `_DISCOVERY_TOOLS`/`_MUTATION_TOOLS` (lines 1147–1149, 1171) and dispatched via the early-return branches above; the registry entries are never reached for them. After these three branches, the function falls through to the six generic registry lookups (`_DISCOVERY_TOOLS.get`, `_MUTATION_TOOLS.get`, `_BLOB_DISCOVERY_TOOLS.get`, `_BLOB_MUTATION_TOOLS.get`, `_SECRET_DISCOVERY_TOOLS.get`, `_SECRET_MUTATION_TOOLS.get`) at lines 1341–1390.
- **`_all_tools` vs `_all_tools_v2` divergence.**
  - `_all_tools` (lines 1201–1208) unions the six sync registries only, with a `len(...) == sum(len(...))` overlap assertion at 1209–1216 and a "cacheable ⊆ discovery" assertion at 1218–1220.
  - `_all_tools_v2` (lines 1401–1409) re-computes the same union but also adds `set(_SESSION_AWARE_TOOL_HANDLERS)`, with a parallel overlap assertion at 1410–1421 and an `iscoroutinefunction` sweep at 1426–1429 (async handlers must be in the v2-only async registry) plus a sync-iscoroutine sweep at 1439–1451. The two aggregates exist side-by-side because v1 is a sync-only registry checkpoint and v2 extends it with the async session-aware tier — but both remain module-level, both run their own assertions, and only `_all_tools_v2` reflects the actual full tool universe today.

---

## `_common.py` (1187 lines)

**Surface.** No external (out-of-package) callers — all uses go through `tools/__init__.py` re-exports. Exports the shared toolkit: `ToolResult` (result wrapper), `ToolHandler` (handler type alias), `RuntimePreflight` (callback type), helpers `_discovery_result`, `_mutation_result`, `_failure_result`, the validation/repair vocabulary (`_GraphRepairSuggestion`, `_RepairToolCall`, `_AffectedConsumer`, `_FullPipelineStatePayload` and its sub-payloads, `_SemanticEdgeContractPayload`), pre-validators (`_prevalidate_source`, `_prevalidate_sink`, `_prevalidate_transform`, `_prevalidate_plugin_options`, `_validate_plugin_name`, `_validate_source_path`, `_validate_sink_path`, `_validate_mutation_arguments`, `_validate_aggregation_trigger`), serialisers (`_serialize_source`, `_serialize_node`, `_serialize_edge`, `_serialize_output`), policy frozensets (`_APPEND_COLLISION_POLICIES`, `_WRITE_COLLISION_POLICIES`, `_FILE_SINK_REPAIR_EXTENSIONS`, `_FILE_SINKS_REQUIRING_COLLISION_POLICY`, `_WEB_ONLY_SOURCE_KEYS`), `_DATA_ERROR_KEY`, `_DEFAULT_SOURCE_VALIDATION_FAILURE`, `_SOURCE_VALIDATION_FAILURE_DESCRIPTION`, the placeholder-masking helper `_mask_pending_interpretation_placeholders_for_authoring_validation`, `diff_states`, `_compute_validation_delta`, `_apply_merge_patch`, `_attach_post_call_hints`, `_credential_wiring_contract_failure`, `_graph_repair_suggestions`, `_duplicate_consumer_repair_suggestions`, `_missing_output_options_repair_error`, `_secret_ref_placement_error`, `_repair_identifier_fragment`, `_reserved_connection_names`, `_unique_name`, `_semantic_contracts_payload`, `_prepend_rejection_entry`, `_vf_destination_note`, `validate_composer_file_sink_collision_policy`.

**Internal layering.** Imports only from `elspeth.web.composer.state` (line 48) plus `web.execution.schemas`, `web.paths`, `web.secrets.ref_policy`, `web.validation`, `core.config`, `core.secrets`, `plugins.infrastructure.{config_base, validation}`, `contracts.freeze`, `web.catalog.protocol`, `web.composer.protocol`. No sibling imports — it is the leaf that every other plane consumes.

**Cross-plane private-symbol coupling.** Imports four private symbols from `elspeth.web.composer.state`: `_coalesce_branch_connections`, `_coalesce_branch_names`, `_serialize_branches`, plus the *contract* types `ValidationEntry`, `ValidationSummary`. Confirms the audit ticket's C5 finding (`_common.py` pulls private serialisation helpers from `composer.state`).

---

## `blobs.py` (1268 lines)

**Surface.** Public-shape: `BlobCreatePayload` (TypedDict), `BlobToolHandler` (type alias), `BlobToolRecord` (TypedDict at line 78), `_BlobQuotaExceededInTxn`, `_BlobUpdateBlockedByActiveRun` (exceptions), `_PreparedBlobCreate` (dataclass). Module state: `_BLOB_QUOTA_BYTES = 500 * 1024 * 1024` (line 242), `_SESSION_BLOB_LOCKS: dict[str, threading.Lock]` (line 636) plus a registry mutex `_SESSION_BLOB_LOCKS_REGISTRY_MUTEX`. Policy frozensets: `_ALLOWED_BLOB_MIME_TYPES`, `_BLOB_QUOTA_MUTATION_TOOLS` (line 1243), `_BLOB_PROVENANCE_MUTATION_TOOLS` (line 1251), `_BLOB_STORE_ONLY_MUTATION_TOOLS` (line 1258). Predicate `is_blob_store_only_mutation_tool` (line 1261). Sync helpers for tests: `_sync_get_blob`, `_sync_get_blob_by_storage_path`, `_sync_list_blobs`. Handlers: `_handle_list_blobs`, `_handle_get_blob_metadata`, `_execute_get_blob_content`, `_execute_create_blob`, `_execute_update_blob`, `_execute_delete_blob`. Plus helpers `_blob_create_payload`, `_blob_row_to_tool_dict`, `_blob_storage_path`, `_check_blob_quota`, `_persist_prepared_blob_create`, `_prepare_blob_create`, `_resolve_blob_quota_bytes`, `_session_blob_lock`, `_verify_blob_content_integrity`.

**Internal layering.** Sibling import only of `_common` (line 68: `ToolResult`, `_discovery_result`, `_failure_result`).

**Cross-plane private-symbol coupling.** Imports five private/internal symbols from `elspeth.web.blobs.service` (lines 51–60): `_ACTIVE_RUN_COMPOSITION_COLUMNS`, `_active_run_pipeline_dict`, `_composition_references_blob`, `_guard_blob_row_literals`, plus public `content_hash` and `sanitize_filename`. Confirmed in `web/blobs/service.py` at lines 56, 78, 154, 242. This is the ticket's **C1** cross-plane coupling: `tools.blobs` reaches into the engine-owned `web.blobs.service` for private active-run / composition / row-guard helpers. Also imports raw SQLAlchemy table objects from `web.sessions.models` (line 75: `blob_run_links_table`, `blobs_table`, `composition_states_table`, `runs_table`).

---

## `generation.py` (1135 lines)

**Surface.** Major handlers: `_execute_preview_pipeline`, `_execute_diff_pipeline`, `_execute_explain_validation_error`, `_execute_get_audit_info`, `_execute_get_plugin_assistance`, `_execute_list_models`, `_handle_get_expression_grammar`, `_handle_get_plugin_schema`, plus `compute_proof_diagnostics` and `get_expression_grammar` (the only non-underscore handler exports). Module state: `_AUTHORING_VALIDATION_COUNTER` (OpenTelemetry counter), `_EXPRESSION_GRAMMAR` (huge string literal), `_BLOCKING_DIAGNOSTIC_CODES`, `_VALIDATION_ERROR_PATTERNS`, `_NUMERIC_VALUE_FIELD_AGGREGATION_PLUGINS`. CSV-inspection helpers: `_sample_csv_rows`, `_source_schema_mode`, `_row_fields_referenced_by_condition`, `_source_field_reaches_connection_without_type_change`, `_value_transform_preserves_field`, `_gate_expression_type_diagnostics_for_observed_csv`, `_numeric_aggregation_diagnostics_for_observed_csv`. Validator-hint helpers: `_augment_with_expected_hint`, `_extract_validator_expected_hint`, `_blocking_diagnostic`, `_serialize_plugin_assistance_example`.

**Internal layering.** Sibling imports of `_common` (line 31), `blobs` (line 39), `sessions` (line 43).

**Cross-plane private-symbol coupling.**
- From `blobs` (lines 39–42): `_sync_get_blob`, `_verify_blob_content_integrity`. Confirms **C4** (generation reaches the blob plane's sync helpers to fetch CSV content during preview's proof phase).
- From `sessions` (line 43): `_authoring_validation_payload` — the validation payload helper lives in the sessions plane but is consumed by the generation plane. Confirms **C6** (sessions-plane private reused by generation).
- From `web.composer.state` (lines 23–30): `_source_options_have_schema`, `_validate_gate_expression` — both underscore-prefixed.

---

## `sessions.py` (986 lines)

**Surface.** Module-level: `ADVISOR_TRIGGER_PROACTIVE_RED_LISTED`, `ADVISOR_TRIGGER_PROACTIVE_SECURITY`, `ADVISOR_TRIGGER_REACTIVE`, `ADVISOR_TRIGGER_VALUES` (advisor-trigger string constants); `RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE`, `RATE_CAP_PER_SESSION_DAY_CODE`, `RATE_CAP_PER_TERM_CODE`; `_FULL_STATE_COMPONENT_ALIASES`, `_FULL_STATE_COMPONENT_ALIAS_SET`. Handler set: `_execute_get_pipeline_state`, `_execute_set_pipeline`, `_handle_set_pipeline`, `_handle_request_interpretation_review`, plus `is_session_aware_tool` (line 979) and the registry `_SESSION_AWARE_TOOL_HANDLERS` (line 974, currently `{"request_interpretation_review": _handle_request_interpretation_review}`). Type aliases `SessionAwareToolHandler` and arguments models `_RequestInterpretationReviewArgumentsModel`. Helpers `_assert_affected_llm_node`, `_authoring_validation_payload`, `_check_interpretation_rate_limits`, `_detect_unresolved_interpretation_placeholders[_typed]`, `_is_full_state_component_alias`, `_serialize_full_pipeline_state`, `_utc_day_start`.

**Internal layering.** Sibling imports of `_common` (line 37, 26 symbols), `blobs` (line 62, four symbols), `sources` (line 68, five symbols). Plus imports `interpretation_sites` from `web.interpretation_state` (line 74 onwards).

**Cross-plane private-symbol coupling.**
- From `blobs` (lines 62–67): `_blob_create_payload`, `_persist_prepared_blob_create`, `_prepare_blob_create`, `_PreparedBlobCreate` — sessions uses the blob plane's create-payload machinery when `set_pipeline` carries an inline-blob source.
- From `sources` (lines 68–74): `_MIME_TO_SOURCE`, `_header_only_inline_csv_conflict`, `_reject_manual_source_blob_ref`, `_resolve_source_blob`, `_ResolvedSourceBlob`. Confirms the audit ticket's bidirectional sessions↔sources/blobs coupling.
- From `web.composer.state` (lines 23–36): private helpers `_batch_aware_placement_error`, `_batch_aware_required_input_fields_error`, `_validate_gate_expression`.
- Sessions also makes the `_MUTATION_TOOLS` invariant explicit in its docstring at line 870 ("dual-registry invariant") — meaning the v1/v2 split is consciously owned but not collapsed.

---

## `sources.py` (607 lines)

**Surface.** `SourceBlobPayload` (TypedDict), `_MIME_TO_SOURCE` (dict). Handlers: `_handle_list_sources`, `_handle_set_source`, `_handle_clear_source`, `_handle_patch_source_options`, `_execute_set_source`, `_execute_clear_source`, `_execute_patch_source_options`, `_execute_set_source_from_blob`, `_execute_inspect_source`. Helpers: `_first_nonempty_csv_row`, `_header_only_inline_csv_conflict`, `_is_header_only_csv`, `_manual_source_blob_ref_error`, `_reject_manual_source_blob_ref`, `_resolve_source_blob`, `_ResolvedSourceBlob`, `_source_blob_payload`.

**Internal layering.** Sibling imports of `_common` (line 33, 12 symbols) and `blobs` (line 47, five symbols).

**Cross-plane private-symbol coupling.** From `blobs` (lines 47–53): `BlobToolRecord`, `_blob_row_to_tool_dict`, `_PreparedBlobCreate`, `_sync_get_blob`, `_verify_blob_content_integrity` — sources plane reaches into blobs plane for row → dict marshalling and sync lookups. Also imports `blobs_table` directly from `web.sessions.models` at line 55.

---

## `transforms.py` (514 lines)

**Surface.** Handlers `_handle_list_transforms`, `_handle_upsert_node`, `_handle_upsert_edge`, `_handle_remove_node`, `_handle_remove_edge`, `_handle_set_metadata`, `_handle_patch_node_options`, plus their `_execute_…` synonyms. Arguments-model classes `_UpsertNodeArgumentsModel`, `_UpsertEdgeArgumentsModel`, `_RemoveByIdArgumentsModel`, `_SetMetadataArgumentsModel`, `_SetMetadataPatchModel`. Policy constant `_NODE_ROUTING_OPTION_PATCH_KEYS`. Helper `_node_routing_option_patch_error`.

**Internal layering.** Sibling import of `_common` only (line 28, 12 symbols). No imports from `blobs`, `sessions`, `sources`.

**Cross-plane private-symbol coupling.** From `web.composer.state` (lines 17–26): `_batch_aware_placement_error`, `_batch_aware_required_input_fields_error`, `_validate_gate_expression` — same trio also imported by `sessions.py`, confirming the audit ticket's note that state-plane privates are widely shared.

---

## `outputs.py` (221 lines)

**Surface.** Handlers `_handle_set_output`, `_handle_remove_output`, `_handle_patch_output_options`, plus the matching `_execute_…` variants. Arguments models `_SetOutputArgumentsModel`, `_RemoveOutputArgumentsModel`.

**Internal layering.** Sibling import of `_common` only (line 19, 11 symbols). No imports from `blobs`, `sessions`, `sources`, `transforms`.

**Cross-plane private-symbol coupling.** None across siblings. Only state-plane import is the public `OutputSpec` + `CompositionState`.

---

## `recipes.py` (162 lines)

**Surface.** Handlers `_execute_list_recipes` and `_execute_apply_pipeline_recipe`. No model classes, no module state.

**Internal layering.** Sibling imports of `_common` (line 25) and `sessions` (line 30).

**Cross-plane private-symbol coupling.** From `sessions` (line 30): `_execute_set_pipeline` — recipe application delegates to the session plane's pipeline-setter. This is a one-direction sibling-private import (recipes → sessions).

---

## `secrets.py` (158 lines)

**Surface.** `SecretToolHandler` (type alias, line at end of import block). Handlers `_handle_list_secret_refs`, `_handle_validate_secret_ref`, `_execute_wire_secret_ref`.

**Internal layering.** Sibling import of `_common` only (line 16, 6 symbols including `_secret_ref_placement_error`).

**Cross-plane private-symbol coupling.** None across siblings; only state-plane public types (`NodeSpec`, `OutputSpec`, `SourceSpec`, `CompositionState`).

---

## `sinks.py` (23 lines)

**Surface.** Single handler `_handle_list_sinks` (line 17). Body is one line: `return _discovery_result(state, catalog.list_sinks())`.

**Internal layering.** Sibling import of `_common` only (line 11: `ToolResult`, `_discovery_result`).

**Cross-plane private-symbol coupling.** None. The smallest plane and the only one with strictly zero cross-plane private coupling.

---

## Cross-cutting concerns

### Discovery / mutation symbol cluster

Six registries with parallel naming, four "is_*" predicates, two aggregate sets, and two assertions. Locations:

| Symbol | File | Line | Purpose |
| --- | --- | --- | --- |
| `_DISCOVERY_TOOLS` | `_dispatch.py` | 1136 | 13 read-only handlers (sync) |
| `_CACHEABLE_DISCOVERY_TOOLS` | `_dispatch.py` | 1152 | `_DISCOVERY_TOOLS` minus `{diff_pipeline, get_pipeline_state, preview_pipeline}` |
| `_MUTATION_TOOLS` | `_dispatch.py` | 1159 | 13 sync mutation handlers |
| `_BLOB_DISCOVERY_TOOLS` | `_dispatch.py` | 1175 | 4 blob/source-inspect handlers |
| `_BLOB_MUTATION_TOOLS` | `_dispatch.py` | 1183 | 5 blob/recipe handlers |
| `_BLOB_QUOTA_MUTATION_TOOLS` | `blobs.py` | 1243 | Which blob mutations consume `max_blob_storage_per_session_bytes` |
| `_BLOB_PROVENANCE_MUTATION_TOOLS` | `blobs.py` | 1251 | Which blob mutations consume `user_message_id` |
| `_BLOB_STORE_ONLY_MUTATION_TOOLS` | `blobs.py` | 1258 | `{create_blob, update_blob, delete_blob}` |
| `_SECRET_DISCOVERY_TOOLS` | `_dispatch.py` | 1191 | 2 secret handlers |
| `_SECRET_MUTATION_TOOLS` | `_dispatch.py` | 1197 | 1 secret handler (`wire_secret_ref`) |
| `_SESSION_AWARE_TOOL_HANDLERS` | `sessions.py` | 974 | 1 async handler (`request_interpretation_review`) |
| `_all_tools` | `_dispatch.py` | 1201 | Union of the six sync registries |
| `_all_tools_v2` | `_dispatch.py` | 1401 | Same union plus `_SESSION_AWARE_TOOL_HANDLERS` |
| `is_discovery_tool` | `_dispatch.py` | 1223 | Membership in `_DISCOVERY_TOOLS ∪ _BLOB_DISCOVERY_TOOLS ∪ _SECRET_DISCOVERY_TOOLS` |
| `is_mutation_tool` | `_dispatch.py` | 1228 | Membership in `_MUTATION_TOOLS ∪ _BLOB_MUTATION_TOOLS ∪ _SECRET_MUTATION_TOOLS` |
| `is_cacheable_discovery_tool` | `_dispatch.py` | 1233 | Membership in `_CACHEABLE_DISCOVERY_TOOLS` |
| `is_session_aware_tool` | `sessions.py` | 979 | Membership in `_SESSION_AWARE_TOOL_HANDLERS` |
| `is_blob_store_only_mutation_tool` | `blobs.py` | 1261 | Membership in `_BLOB_STORE_ONLY_MUTATION_TOOLS` |

Externally consumed in `src/elspeth/web/composer/service.py` at lines 549, 566, 596, 2270, 2314, 2380, 2413, 2480, 2519, 2528, 2938, 3104, 3297, 3309, and in `src/elspeth/web/composer/progress.py` at 404, 434, 457. Predicates are spread across three files (`_dispatch.py`, `blobs.py`, `sessions.py`) and the registry constants used by `_dispatch.execute_tool` live in two files (`_dispatch.py` and `blobs.py`).

### Registry/dispatch divergence (`_all_tools` vs `_all_tools_v2`)

- `_all_tools` (lines 1201–1216): union of the six **sync** registries, with overlap assertion. Followed at 1218–1220 by the "cacheable ⊆ discovery" check.
- `_all_tools_v2` (lines 1401–1421): unions the same six sync registries **plus** `_SESSION_AWARE_TOOL_HANDLERS`, with a stricter assertion message that names every constituent registry.
- Both aggregates execute at import time. Neither is consumed by runtime code — they exist purely as assertion namespaces. The v1 form is kept alive (not collapsed into v2) and re-exported through `tools/__init__.py:319-320`.
- Followed at 1426–1429 by an `iscoroutinefunction` sweep over `_SESSION_AWARE_TOOL_HANDLERS` (handlers must be async) and at 1439–1451 by the reverse sweep over the six sync registries (handlers must not be async).

### Three remaining hardcoded dispatch branches in `execute_tool`

Lines 1301 (`preview_pipeline`), 1313 (`diff_pipeline`), 1326 (`set_pipeline`). Each is in `_DISCOVERY_TOOLS` or `_MUTATION_TOOLS` (lines 1147–1149, 1171) but routes via an early-return branch because of extended kwargs:

- `preview_pipeline` needs `runtime_preflight`, `session_engine`, `session_id`.
- `diff_pipeline` needs `baseline`, `current_validation`.
- `set_pipeline` needs `session_engine`, `session_id`, `user_message_id`, `max_blob_storage_per_session_bytes`.

The blob and secret registries already encode their extended-kwarg shape generically (lines 1352–1390): blob handlers always receive `session_engine` + `session_id`, plus optional `max_blob_storage_per_session_bytes` and `user_message_id` gated by membership in `_BLOB_QUOTA_MUTATION_TOOLS` and `_BLOB_PROVENANCE_MUTATION_TOOLS`. Secret handlers always receive `secret_service` + `user_id`. The three discovery/mutation tools above do not share a kwarg shape and therefore have not been folded into the registry dispatch.

### Cross-plane private-symbol coupling (consolidated against audit-ticket cases)

| Audit case | Importer | Source | Symbols |
| --- | --- | --- | --- |
| **C1** | `tools/blobs.py:51` | `web.blobs.service` | `_ACTIVE_RUN_COMPOSITION_COLUMNS`, `_active_run_pipeline_dict`, `_composition_references_blob`, `_guard_blob_row_literals` |
| **C4** | `tools/generation.py:39` | `tools/blobs.py` | `_sync_get_blob`, `_verify_blob_content_integrity` |
| **C5** | `tools/_common.py:48` | `web.composer.state` | `_coalesce_branch_connections`, `_coalesce_branch_names`, `_serialize_branches` |
| **C6** | `tools/generation.py:43` | `tools/sessions.py` | `_authoring_validation_payload` |
| (not in ticket) | `tools/sessions.py:62` | `tools/blobs.py` | `_blob_create_payload`, `_persist_prepared_blob_create`, `_prepare_blob_create`, `_PreparedBlobCreate` |
| (not in ticket) | `tools/sessions.py:68` | `tools/sources.py` | `_MIME_TO_SOURCE`, `_header_only_inline_csv_conflict`, `_reject_manual_source_blob_ref`, `_resolve_source_blob`, `_ResolvedSourceBlob` |
| (not in ticket) | `tools/sources.py:47` | `tools/blobs.py` | `BlobToolRecord`, `_blob_row_to_tool_dict`, `_PreparedBlobCreate`, `_sync_get_blob`, `_verify_blob_content_integrity` |
| (not in ticket) | `tools/recipes.py:30` | `tools/sessions.py` | `_execute_set_pipeline` |
| (not in ticket) | `tools/_dispatch.py:26` | `tools/blobs.py` | `_BLOB_PROVENANCE_MUTATION_TOOLS`, `_BLOB_QUOTA_MUTATION_TOOLS` |
| (not in ticket) | `tools/_dispatch.py:62` | `tools/sessions.py` | `_SESSION_AWARE_TOOL_HANDLERS` |
| (not in ticket) | `tools/transforms.py:17` and `tools/sessions.py:23` and `tools/generation.py:23` | `web.composer.state` | `_batch_aware_placement_error`, `_batch_aware_required_input_fields_error`, `_validate_gate_expression` (sessions adds the trio; transforms adds the trio; generation adds `_source_options_have_schema` + `_validate_gate_expression`) |

Direction-of-coupling summary (plane → plane):

```
_common      ← every other plane (one-way)
blobs        ← sources, sessions, generation, _dispatch
sources      ← sessions
sessions     ← recipes, generation, _dispatch
generation   ← _dispatch
sinks        ← _dispatch
secrets      ← _dispatch
outputs      ← _dispatch
transforms   ← _dispatch
recipes      ← _dispatch
state(L1-ish)← _common, transforms, sessions, generation (private symbols)
blobs.service← blobs (private symbols, ticket C1)
```

The audit ticket's "C1 / C4 / C5 / C6" enumeration is accurate but partial: there are at least six additional cross-plane private-symbol import sites inside `tools/` itself (sessions↔blobs, sessions↔sources, sources↔blobs, recipes↔sessions, dispatch↔blobs, dispatch↔sessions), plus the wider state-plane private surface used by transforms, sessions, and generation.
