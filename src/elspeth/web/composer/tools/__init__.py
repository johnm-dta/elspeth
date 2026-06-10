"""Composer tool surface — discovery and mutation tools for the LLM composer.

Decomposed package. Plane modules:

- ``_common`` — ``ToolResult``, response helpers, repair-suggestion synthesis,
  validation helpers, base serialisation, type aliases. The shared toolkit
  every other plane imports from.
- ``blobs`` — session-scoped binary blob storage handlers.
- ``sources`` — source-spec mutation handlers and source-from-blob bridge.
- ``transforms`` — node, edge, metadata graph-mutation handlers, plus sink
  discovery (sinks are terminal transforms; no separate plane).
- ``outputs`` — output (sink-instance) mutation handlers.
- ``recipes`` — pipeline-recipe discovery and application handlers.
- ``secrets`` — secret-reference discovery, validation, and wiring handlers.
- ``sessions`` — pipeline-state and interpretation-review handlers.
- ``generation`` — preview, diff, explain, plugin schema, proof diagnostics.
- ``_dispatch`` — merged registries, ``execute_tool``, predicates, definitions.

This ``__init__`` re-exports only the names that have at least one external
importer (``from elspeth.web.composer.tools import X``). Every entry below is
contractual surface this team commits to defending. Adding a name here MUST
be motivated by an actual cross-package consumer — internal callers within
``elspeth.web.composer.tools.*`` import from their sibling submodules
directly, not via this facade.

CLOSED LIST. **Before re-adding a name here to silence an import error,
fix the importer instead** — sibling submodules under
``elspeth.web.composer.tools.*`` must import from each other directly
(``from elspeth.web.composer.tools.sessions import _execute_set_pipeline``),
NOT via this facade. Re-adding a name to chase down an import error
re-inflates the surface the underscore-rename + prune commits (67df87181,
4b66ad52c) deliberately collapsed. Extension of this list requires a
design-review decision attached to a cross-package consumer; the
regression test catches dead entries but cannot catch
deliberately-excluded names that quietly grew a new consumer.

The regression test
``tests/unit/web/composer/test_tools_facade_surface.py`` fails the build if
any ``__all__`` entry becomes dead.
"""

from elspeth.web.composer.tools._common import (
    _DATA_ERROR_KEY,
    RuntimePreflight,
    ToolContext,
    ToolResult,
    _apply_merge_patch,
    _attach_post_call_hints,
    _compute_validation_delta,
    _credential_wiring_contract_failure,
    _failure_result,
    _prevalidate_plugin_options,
    diff_states,
    validate_composer_file_sink_collision_policy,
)
from elspeth.web.composer.tools._dispatch import (
    _inject_prior_validation,
    execute_tool,
    get_tool_definitions,
)
from elspeth.web.composer.tools._registry import (
    _BLOB_DISCOVERY_TOOLS,
    _BLOB_MUTATION_TOOLS,
    _CACHEABLE_DISCOVERY_TOOL_NAMES,
    _DISCOVERY_TOOLS,
    _MUTATION_TOOLS,
    _SECRET_DISCOVERY_TOOLS,
    _SECRET_MUTATION_TOOLS,
)
from elspeth.web.composer.tools.blobs import (
    _ALLOWED_BLOB_MIME_TYPES,
    _execute_create_blob,
    _execute_update_blob,
    _persist_prepared_blob_create,
    _prepare_blob_create,
    _session_blob_lock,
    _sync_get_blob_by_storage_path,
    _sync_list_blobs,
)
from elspeth.web.composer.tools.discovery import (
    is_blob_store_only_mutation_tool,
    is_cacheable_discovery_tool,
    is_discovery_tool,
    is_mutation_tool,
    is_session_aware_tool,
)
from elspeth.web.composer.tools.generation import (
    _BLOCKING_DIAGNOSTIC_CODES,
    _blocking_diagnostic,
    _execute_preview_pipeline,
    compute_proof_diagnostics,
    get_expression_grammar,
)
from elspeth.web.composer.tools.outputs import (
    _execute_patch_output_options,
    _execute_set_output,
)
from elspeth.web.composer.tools.sessions import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_DETERMINISTIC_EARLY,
    ADVISOR_TRIGGER_DETERMINISTIC_END,
    ADVISOR_TRIGGER_VALUES,
    RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE,
    RATE_CAP_PER_SESSION_DAY_CODE,
    RATE_CAP_PER_TERM_CODE,
    _check_interpretation_rate_limits,
    _detect_unresolved_interpretation_placeholders_typed,
    _execute_apply_pipeline_recipe,
    _execute_set_pipeline,
    _handle_request_interpretation_review,
    _utc_day_start,
)
from elspeth.web.composer.tools.sources import (
    _execute_patch_source_options,
    _execute_set_source,
    _execute_set_source_from_blob,
    _resolve_source_blob,
    _ResolvedSourceBlob,
)
from elspeth.web.composer.tools.transforms import (
    _execute_patch_node_options,
)

__all__ = [
    "ADVISOR_TRIGGER_DETERMINISTIC_EARLY",
    "ADVISOR_TRIGGER_DETERMINISTIC_END",
    "ADVISOR_TRIGGER_VALUES",
    "RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE",
    "RATE_CAP_PER_SESSION_DAY_CODE",
    "RATE_CAP_PER_TERM_CODE",
    "_ALLOWED_BLOB_MIME_TYPES",
    "_BLOB_DISCOVERY_TOOLS",
    "_BLOB_MUTATION_TOOLS",
    "_BLOCKING_DIAGNOSTIC_CODES",
    "_CACHEABLE_DISCOVERY_TOOL_NAMES",
    "_DATA_ERROR_KEY",
    "_DISCOVERY_TOOLS",
    "_MUTATION_TOOLS",
    "_SECRET_DISCOVERY_TOOLS",
    "_SECRET_MUTATION_TOOLS",
    "_SESSION_AWARE_TOOL_HANDLERS",
    "RuntimePreflight",
    "ToolContext",
    "ToolResult",
    "_ResolvedSourceBlob",
    "_apply_merge_patch",
    "_attach_post_call_hints",
    "_blocking_diagnostic",
    "_check_interpretation_rate_limits",
    "_compute_validation_delta",
    "_credential_wiring_contract_failure",
    "_detect_unresolved_interpretation_placeholders_typed",
    "_execute_apply_pipeline_recipe",
    "_execute_create_blob",
    "_execute_patch_node_options",
    "_execute_patch_output_options",
    "_execute_patch_source_options",
    "_execute_preview_pipeline",
    "_execute_set_output",
    "_execute_set_pipeline",
    "_execute_set_source",
    "_execute_set_source_from_blob",
    "_execute_update_blob",
    "_failure_result",
    "_handle_request_interpretation_review",
    "_inject_prior_validation",
    "_persist_prepared_blob_create",
    "_prepare_blob_create",
    "_prevalidate_plugin_options",
    "_resolve_source_blob",
    "_session_blob_lock",
    "_sync_get_blob_by_storage_path",
    "_sync_list_blobs",
    "_utc_day_start",
    "compute_proof_diagnostics",
    "diff_states",
    "execute_tool",
    "get_expression_grammar",
    "get_tool_definitions",
    "is_blob_store_only_mutation_tool",
    "is_cacheable_discovery_tool",
    "is_discovery_tool",
    "is_mutation_tool",
    "is_session_aware_tool",
    "validate_composer_file_sink_collision_policy",
]
