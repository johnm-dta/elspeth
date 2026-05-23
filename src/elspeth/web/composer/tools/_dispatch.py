"""Composer dispatch + registry — the tool execution surface (merges every plane's handlers)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, cast

from sqlalchemy import Engine

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.state import (
    CompositionState,
    ValidationSummary,
)
from elspeth.web.composer.tools._common import (
    _DEFAULT_SOURCE_VALIDATION_FAILURE,
    _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
    RuntimePreflight,
    ToolHandler,
    ToolResult,
    _failure_result,
    build_plugin_schemas_for_failure,
    should_augment_with_plugin_schemas,
)
from elspeth.web.composer.tools.blobs import (
    _BLOB_PROVENANCE_MUTATION_TOOLS,
    _BLOB_QUOTA_MUTATION_TOOLS,
    BlobToolHandler,
    _execute_create_blob,
    _execute_delete_blob,
    _execute_get_blob_content,
    _execute_update_blob,
    _handle_get_blob_metadata,
    _handle_list_blobs,
)
from elspeth.web.composer.tools.generation import (
    _execute_diff_pipeline,
    _execute_explain_validation_error,
    _execute_get_audit_info,
    _execute_get_plugin_assistance,
    _execute_list_models,
    _execute_preview_pipeline,
    _handle_get_expression_grammar,
    _handle_get_plugin_schema,
)
from elspeth.web.composer.tools.outputs import (
    _handle_patch_output_options,
    _handle_remove_output,
    _handle_set_output,
)
from elspeth.web.composer.tools.recipes import (
    _execute_apply_pipeline_recipe,
    _execute_list_recipes,
)
from elspeth.web.composer.tools.secrets import (
    SecretToolHandler,
    _execute_wire_secret_ref,
    _handle_list_secret_refs,
    _handle_validate_secret_ref,
)
from elspeth.web.composer.tools.sessions import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_VALUES,
    _execute_get_pipeline_state,
    _execute_set_pipeline,
    _handle_set_pipeline,
)
from elspeth.web.composer.tools.sinks import (
    _handle_list_sinks,
)
from elspeth.web.composer.tools.sources import (
    _execute_inspect_source,
    _execute_set_source_from_blob,
    _handle_clear_source,
    _handle_list_sources,
    _handle_patch_source_options,
    _handle_set_source,
)
from elspeth.web.composer.tools.transforms import (
    _handle_list_transforms,
    _handle_patch_node_options,
    _handle_remove_edge,
    _handle_remove_node,
    _handle_set_metadata,
    _handle_upsert_edge,
    _handle_upsert_node,
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 39 tools: 13 discovery + 13 mutation + 9 blob tools + 3 secret
    tools + 1 advisor tool. ``request_advisor_hint`` is the only tool that
    is filtered out of the LLM-visible list when the operator's
    ``composer_advisor_enabled`` flag is False (the default) — see
    ``ComposerServiceImpl._get_litellm_tools``.

    The skill at ``src/elspeth/web/composer/skills/pipeline_composer.md``
    enumerates the same tool set in its Foundation-knowledge section
    (under "## CRITICAL: Tool Schema Availability"). The drift gate
    ``TestComposerToolNameDrift::test_skill_tool_inventory_matches_get_tool_definitions``
    in ``tests/unit/web/composer/test_skill_drift.py`` enforces equality
    between the runtime list returned here and the skill's bulleted
    categories — adding a tool without updating both sides fails CI.
    """
    return [
        # Discovery tools
        {
            "name": "list_sources",
            "description": "List available source plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_transforms",
            "description": "List available transform plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_sinks",
            "description": "List available sink plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plugin_schema",
            "description": "Get the full configuration schema for a plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin type.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv').",
                    },
                },
                "required": ["plugin_type", "name"],
            },
        },
        {
            "name": "get_expression_grammar",
            "description": "Get the gate expression syntax reference.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Mutation tools
        {
            "name": "set_source",
            "description": "Set or replace the pipeline source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Source plugin name."},
                    "on_success": {
                        "type": "string",
                        "description": (
                            "Connection-name string this source PUBLISHES. Some downstream consumer "
                            "(transform 'input' or output 'sink_name') MUST equal this value for wiring "
                            "to resolve. The runtime matches strings, not graph topology — pick any "
                            "name unique within the pipeline; it does not need to be the downstream "
                            "node's id."
                        ),
                        "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
                    },
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "on_validation_failure": {
                        "type": "string",
                        "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                    },
                },
                "required": ["plugin", "on_success", "options", "on_validation_failure"],
            },
        },
        {
            "name": "upsert_node",
            "description": (
                "Add or update a pipeline node. "
                "Fields are node_type-dependent: "
                "transform/aggregation use plugin+options; "
                "gate uses condition+routes (or fork_to); "
                "coalesce uses branches+policy+merge. "
                "Omit fields that don't apply to your node_type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node identifier."},
                    "node_type": {
                        "type": "string",
                        "enum": ["transform", "gate", "aggregation", "coalesce"],
                    },
                    "plugin": {
                        "type": ["string", "null"],
                        "description": "Plugin name. Required for transform/aggregation. Null for gate/coalesce.",
                    },
                    "input": {
                        "type": "string",
                        "description": (
                            "Connection-name string this node CONSUMES. MUST equal the value of some "
                            "upstream's on_success (or routes value, or on_error) field. NOT the upstream "
                            "node's id — connections are matched by string, not by graph topology. "
                            "Example: if source.on_success='raw_url_rows', this node sets input='raw_url_rows'."
                        ),
                        "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
                    },
                    "on_success": {
                        "type": ["string", "null"],
                        "description": (
                            "Output connection. Required for transform/aggregation/coalesce. Null for "
                            "gates (routing is via condition/routes). When set, this is the connection-name "
                            "string the node PUBLISHES — some downstream input/sink_name MUST equal this "
                            "value. The runtime matches strings, not topology."
                        ),
                        "examples": ["fetched_text", "scored_rows", "lines_out"],
                    },
                    "on_error": {"type": ["string", "null"], "description": "Error output connection (transform/aggregation only)."},
                    "options": {"type": "object", "description": "Plugin-specific config (transform/aggregation only)."},
                    "condition": {"type": ["string", "null"], "description": "Boolean expression (gate only). Evaluated per row."},
                    "routes": {
                        "type": ["object", "null"],
                        "description": (
                            "Route mapping {true: sink_or_connection_or_discard, false: sink_or_connection_or_discard} "
                            "(gate only, mutually exclusive with fork_to). Use 'discard' to drop that route with "
                            "an audited gate_discarded terminal outcome."
                        ),
                    },
                    "fork_to": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Fork destinations — row is copied to all listed paths (gate only, mutually exclusive with routes).",
                    },
                    "branches": {
                        "type": ["array", "object", "null"],
                        "items": {"type": "string"},
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Branches to merge (coalesce only). Use list form when branch identity and input "
                            "connection are the same, or object form {branch_name: input_connection} when a "
                            "branch flows through transforms before coalescing."
                        ),
                    },
                    "policy": {"type": ["string", "null"], "description": "Merge trigger policy (coalesce only)."},
                    "merge": {"type": ["string", "null"], "description": "Field merge strategy (coalesce only)."},
                    "trigger": {
                        "type": ["object", "null"],
                        "description": "Optional early batch trigger config (aggregation only). Omit, null, or {} for end-of-source-only aggregation.",
                        "additionalProperties": False,
                        "properties": {
                            "count": {
                                "type": ["integer", "null"],
                                "minimum": 1,
                                "description": "Flush after this many accepted rows.",
                            },
                            "timeout_seconds": {
                                "type": ["number", "null"],
                                "exclusiveMinimum": 0,
                                "description": "Flush after this many seconds since the first accepted row.",
                            },
                            "condition": {
                                "type": ["string", "null"],
                                "description": "Boolean expression over row['batch_count'] and row['batch_age_seconds']; do not use end_of_source here.",
                            },
                        },
                    },
                    "output_mode": {
                        "type": ["string", "null"],
                        "enum": ["passthrough", "transform", None],
                        "description": "Aggregation output mode (aggregation only). Defaults to 'transform' if omitted.",
                    },
                    "expected_output_count": {
                        "type": ["integer", "null"],
                        "description": "Expected number of output rows from aggregation (aggregation only). Optional; omit when output count depends on group_by distinct values.",
                    },
                },
                "required": ["id", "node_type", "input"],
            },
        },
        {
            "name": "upsert_edge",
            "description": (
                "Add or update a connection between nodes. When the edge targets a sink, "
                "this also updates the source/node routing field used by runtime "
                "(on_success, on_error, gate routes, or fork destinations)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique edge identifier."},
                    "from_node": {"type": "string", "description": "Source node ID or 'source'."},
                    "to_node": {"type": "string", "description": "Destination node ID or sink name."},
                    "edge_type": {
                        "type": "string",
                        "enum": ["on_success", "on_error", "route_true", "route_false", "fork"],
                    },
                    "label": {"type": ["string", "null"], "description": "Display label."},
                },
                "required": ["id", "from_node", "to_node", "edge_type"],
                "examples": [
                    {
                        "id": "e_judge_layers_error",
                        "from_node": "judge_layers",
                        "to_node": "llm_failures",
                        "edge_type": "on_error",
                        "label": "LLM failures",
                    }
                ],
            },
        },
        {
            "name": "remove_node",
            "description": "Remove a node and all its edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "remove_edge",
            "description": "Remove an edge by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Edge ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "set_metadata",
            "description": "Update pipeline metadata (name and description only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "description": "Partial metadata update. Only included fields are changed.",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["patch"],
            },
        },
        {
            "name": "set_output",
            "description": "Add or replace a pipeline output (sink).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {
                        "type": "string",
                        "description": (
                            "Sink name. This string is BOTH the sink's identifier (used by "
                            "patch_output_options/remove_output) AND the connection-name the sink "
                            "consumes — it MUST equal some upstream's on_success value. Pick a name "
                            "describing the data being written; it does not need to match an upstream "
                            "node's id."
                        ),
                        "examples": ["lines_out", "scored_results", "errors_quarantine"],
                    },
                    "plugin": {"type": "string", "description": "Sink plugin name (e.g. 'csv', 'json')."},
                    "options": {
                        "type": "object",
                        "description": (
                            "Plugin-specific config. For csv/json file sinks in runnable web pipelines, "
                            "include path, schema, and explicit collision_policy."
                        ),
                    },
                    "on_write_failure": {
                        "type": "string",
                        "description": "How to handle per-row write failures. Use 'discard' to drop with audit record, or a sink name (e.g. 'results_failures') to divert failed rows to that failsink.",
                        "default": "discard",
                    },
                },
                "required": ["sink_name", "plugin", "options"],
            },
        },
        {
            "name": "remove_output",
            "description": "Remove a pipeline output (sink) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {"type": "string", "description": "Sink name to remove."},
                },
                "required": ["sink_name"],
            },
        },
        {
            "name": "patch_source_options",
            "description": "Apply a shallow merge-patch to the current source options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "description": "Merge-patch to apply to source options.",
                    },
                },
                "required": ["patch"],
            },
        },
        {
            "name": "patch_node_options",
            "description": "Apply a shallow merge-patch to a node's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged. "
            "Do not use this for node routing fields such as on_success/on_error/input/routes; "
            "use upsert_edge or upsert_node for routing edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of the node to patch.",
                    },
                    "patch": {
                        "type": "object",
                        "description": (
                            "Merge-patch to apply to plugin options only. "
                            "Node-level routing fields such as on_success, on_error, input, routes, "
                            "and fork_to are siblings of options; edit them with upsert_edge or upsert_node."
                        ),
                    },
                },
                "required": ["node_id", "patch"],
            },
        },
        {
            "name": "patch_output_options",
            "description": "Apply a shallow merge-patch to an output's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {
                        "type": "string",
                        "description": "Name of the output (sink) to patch.",
                    },
                    "patch": {
                        "type": "object",
                        "description": "Merge-patch to apply to output options.",
                    },
                },
                "required": ["sink_name", "patch"],
            },
        },
        {
            "name": "set_pipeline",
            "description": "Atomically replace the entire pipeline. Provide the "
            "complete source, nodes, edges, outputs, and metadata in one call. "
            "This is more efficient than calling set_source + upsert_node + "
            "upsert_edge + set_output sequentially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "object",
                        "description": (
                            "Source configuration: {plugin, on_success, options?, on_validation_failure?, blob_id?, inline_blob?}. "
                            "Use blob_id to bind an already uploaded session blob, or inline_blob to "
                            "materialize user-provided literal data while atomically setting the full pipeline."
                        ),
                        "properties": {
                            "plugin": {"type": "string"},
                            "blob_id": {
                                "type": "string",
                                "description": (
                                    "Existing ready session blob ID to bind as this source. "
                                    "The tool resolves path/blob_ref authoritatively exactly like set_source_from_blob."
                                ),
                            },
                            "options": {
                                "type": "object",
                                "description": (
                                    "Plugin-specific source config. Required by most file/data sources even though "
                                    "the schema leaves it optional so the handler can return plugin-specific repair "
                                    "feedback instead of a generic missing-argument error."
                                ),
                            },
                            "on_success": {
                                "type": "string",
                                "description": (
                                    "Connection-name string the source PUBLISHES. Some downstream "
                                    "consumer (node 'input' or output 'sink_name') MUST equal this. "
                                    "Connections match by string, not by node id."
                                ),
                                "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
                            },
                            "on_validation_failure": {
                                "type": "string",
                                "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                            },
                            "inline_blob": {
                                "type": "object",
                                "description": (
                                    "Optional inline source content to create as a session blob before binding the source. "
                                    "Fields mirror create_blob: filename, mime_type, content, and optional description."
                                ),
                                "properties": {
                                    "filename": {"type": "string"},
                                    "mime_type": {
                                        "type": "string",
                                        "enum": [
                                            "text/plain",
                                            "application/json",
                                            "text/csv",
                                            "application/x-jsonlines",
                                            "application/jsonl",
                                            "text/jsonl",
                                        ],
                                    },
                                    "content": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["filename", "mime_type", "content"],
                            },
                        },
                        "required": ["plugin", "on_success"],
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "node_type": {"type": "string"},
                                "plugin": {"type": "string"},
                                "input": {
                                    "type": "string",
                                    "description": (
                                        "Connection-name string this node CONSUMES. MUST equal some "
                                        "upstream's on_success/routes value/on_error. NOT the upstream "
                                        "node's id. If source.on_success='raw_url_rows', this node sets "
                                        "input='raw_url_rows'."
                                    ),
                                    "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
                                },
                                "on_success": {
                                    "type": "string",
                                    "description": (
                                        "Connection-name string this node PUBLISHES (transform/aggregation/"
                                        "coalesce). Some downstream input/sink_name MUST equal this. Omit "
                                        "for gates (routing is via condition+routes)."
                                    ),
                                    "examples": ["fetched_text", "scored_rows", "lines_out"],
                                },
                                "on_error": {"type": "string"},
                                "options": {"type": "object"},
                                "condition": {"type": "string"},
                                "routes": {
                                    "type": "object",
                                    "description": (
                                        "Gate route mapping to sink names, downstream connection names, 'fork', or "
                                        "'discard' for an audited terminal drop."
                                    ),
                                },
                                "fork_to": {"type": "array", "items": {"type": "string"}},
                                "branches": {
                                    "type": ["array", "object"],
                                    "items": {"type": "string"},
                                    "additionalProperties": {"type": "string"},
                                },
                                "policy": {"type": "string"},
                                "merge": {"type": "string"},
                                "trigger": {"type": "object"},
                                "output_mode": {"type": "string"},
                                "expected_output_count": {"type": "integer"},
                            },
                            "required": ["id", "node_type", "input"],
                        },
                        "description": "Array of node specs: [{id, input, plugin?, node_type, options?, on_success?, on_error?, condition?, routes?, fork_to?, branches?, policy?, merge?, trigger?, output_mode?, expected_output_count?}]",
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "from_node": {"type": "string"},
                                "to_node": {"type": "string"},
                                "edge_type": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["id", "from_node", "to_node", "edge_type"],
                        },
                        "description": "Array of edge specs: [{id, from_node, to_node, edge_type}]",
                    },
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sink_name": {
                                    "type": "string",
                                    "description": (
                                        "Sink name. BOTH the sink's identifier AND the connection-name "
                                        "the sink consumes — it MUST equal some upstream's on_success "
                                        "value. Pick a descriptive name; it does not need to match an "
                                        "upstream node's id."
                                    ),
                                    "examples": ["lines_out", "scored_results", "errors_quarantine"],
                                },
                                "plugin": {"type": "string"},
                                "options": {
                                    "type": "object",
                                    "description": (
                                        "Plugin-specific sink config. For csv/json file sinks in runnable web "
                                        "pipelines, include path, schema, and explicit collision_policy."
                                    ),
                                },
                                "on_write_failure": {"type": "string"},
                            },
                            "required": ["sink_name", "plugin"],
                            "examples": [
                                {
                                    "sink_name": "results",
                                    "plugin": "json",
                                    "options": {
                                        "path": "outputs/results.json",
                                        "schema": {"mode": "observed"},
                                        "collision_policy": "auto_increment",
                                    },
                                    "on_write_failure": "discard",
                                }
                            ],
                        },
                        "description": (
                            "Array of output specs: [{sink_name, plugin, options, on_write_failure?}]. "
                            "For csv/json file sinks in runnable web pipelines, options must include "
                            "path, schema, and explicit collision_policy."
                        ),
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Pipeline metadata: {name?, description?}",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["source", "nodes", "edges", "outputs"],
            },
        },
        # Source-reset and validation-explanation tools.
        {
            "name": "clear_source",
            "description": "Remove the source from the pipeline composition state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "explain_validation_error",
            "description": "Get a human-readable explanation of a validation error "
            "with suggested fixes. Pass the exact error text from a validation result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_text": {
                        "type": "string",
                        "description": "The validation error message to explain.",
                    },
                },
                "required": ["error_text"],
            },
        },
        {
            "name": "request_advisor_hint",
            "description": (
                "ESCAPE HATCH — call when one of the declared trigger criteria applies: "
                "reactive validation-loop recovery after two or more unchanged failures, "
                "proactive security/safety wiring review before `set_pipeline`, or "
                "proactive red-listed plugin review before `set_pipeline`. The proactive "
                "security trigger covers content moderation, prompt-injection defence, "
                "secret routing, PII/regulatory sinks, and externally fetched content "
                "flowing toward LLMs. Forwards your problem statement and context to a "
                "frontier model and returns guidance text. The reply is ADVICE, not "
                "configuration — you must still call the appropriate mutation tool "
                "yourself to apply any change. Budget is finite (sized per compose "
                "request, not per session lifetime) and exhausting it returns a "
                "structured error rather than crashing — inspect budget_remaining "
                "in each response. Do NOT call this tool in a loop, do NOT use it "
                "as a substitute for reading validator output. Disabled by default; "
                "only available when the operator has explicitly enabled it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger": {
                        "type": "string",
                        "enum": list(ADVISOR_TRIGGER_VALUES),
                        "description": (
                            "Why this advisor call is allowed. Use reactive_validation_loop "
                            "only after the recovery sequence and at least two unchanged "
                            "validator failures. Use proactive_security_safety before "
                            "set_pipeline for security/safety-sensitive flows. Use "
                            "proactive_red_listed_plugin before set_pipeline when the plan "
                            "uses a red-listed plugin such as llm, database, dataverse, "
                            "Azure safety transforms, RAG retrieval, or Chroma sinks."
                        ),
                    },
                    "problem_summary": {
                        "type": "string",
                        "description": (
                            "Your own statement of what you are trying to do and "
                            "why you are stuck. One or two sentences. Be specific: "
                            "'I cannot get llm transform options to validate against "
                            "the Azure provider schema' is useful; 'help' is not."
                        ),
                    },
                    "recent_errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The last validator error messages verbatim, most recent first. Include up to 5; do not paraphrase."
                        ),
                    },
                    "attempted_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "What you have already tried, one item per attempt. "
                            "Include the tool name and a one-line summary of the "
                            "argument shape. The advisor uses this to avoid "
                            "suggesting things you have already ruled out."
                        ),
                    },
                    "schema_excerpt": {
                        "type": "string",
                        "description": (
                            "Optional — the relevant plugin schema snippet you are "
                            "working against, as returned by `get_plugin_schema`. "
                            "Including this lets the advisor give field-level "
                            "guidance grounded in the exact contract."
                        ),
                    },
                },
                "required": ["trigger", "problem_summary", "recent_errors", "attempted_actions"],
            },
        },
        {
            "name": "get_plugin_assistance",
            "description": (
                "Retrieve plugin-owned guidance for a source, transform, or sink. "
                "Two modes by ``issue_code``:\n"
                "  * Omit ``issue_code`` (or pass null) to get discovery-time guidance "
                "    — a summary of the plugin and composer_hints. (The same hints "
                "    are also carried on list_sources / list_transforms / list_sinks / "
                "    get_plugin_schema responses; this tool is the explicit path.)\n"
                "  * Pass an ``issue_code`` (validators emit these as requirement_code "
                "    on semantic_contracts entries) to get failure-time guidance — "
                "    summary, suggested_fixes, and example before/after configurations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin family. 'source', 'transform', or 'sink'.",
                    },
                    "plugin_name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv', 'web_scrape', 'database').",
                    },
                    "issue_code": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional. Stable issue identifier owned by the plugin "
                            "for failure-time guidance. Omit or pass null for "
                            "discovery-time guidance."
                        ),
                    },
                },
                "required": ["plugin_type", "plugin_name"],
            },
        },
        {
            "name": "list_models",
            "description": "List available LLM model identifiers. Without a provider "
            "filter, returns provider names and counts. With a provider filter, "
            "returns matching model IDs (capped at limit). For provider='openrouter/' "
            "the returned slugs are normalised to OpenRouter's HTTP API form "
            "(without the litellm-internal 'openrouter/' routing prefix) — these "
            "are the values to put directly in `model:`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider prefix to filter by (e.g. 'openrouter/', 'azure/'). "
                        "Omit to get a provider summary instead of individual models.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max models to return (default 50).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_audit_info",
            "description": (
                "Return facts about ELSPETH's Landscape audit trail. Call this BEFORE "
                "answering any user question that mentions audit logging, audit "
                "database, SQLite/Postgres audit, audit backend, audit export, "
                "Landscape, or 'how do I record what the pipeline did'. Audit is "
                "mandatory and operator-managed; the composer cannot configure the "
                "backend (security boundary — see yaml_generator.py:179, fix S1). "
                "Returns enabled status, composer_modifiable flag, and a canonical "
                "summary to paraphrase. Does NOT return the audit URL/path/DSN — "
                "that is operator-internal and intentionally not surfaced to the LLM."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "preview_pipeline",
            "description": "Preview the current pipeline configuration — returns "
            "validation status, source summary, and node/output overview "
            "without executing. Use this to confirm the pipeline is set up "
            "correctly before running.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_pipeline_state",
            "description": "Inspect the full current pipeline state including all "
            "options for source, nodes, and outputs. Use this during correction "
            "loops to see what is currently configured before patching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": (
                            "Optional: return only one component — 'source', a node ID, or an output name. "
                            "Accepted full-state aliases: omit component, pass 'full', 'all', 'pipeline', "
                            "or pass the empty string."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "diff_pipeline",
            "description": "Show what changed since the session was loaded or created. "
            "Returns added, removed, and modified nodes/edges/outputs, "
            "plus warnings introduced or resolved.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Blob tools
        {
            "name": "list_blobs",
            "description": "List uploaded/created files (blobs) in this session with metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_blob_metadata",
            "description": "Get metadata for a specific blob (file) by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID."},
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "set_source_from_blob",
            "description": "Wire a blob as the pipeline source. Resolves the blob's storage path internally and infers the source plugin from its MIME type. "
            "Use 'options' for plugin-specific config (e.g., 'column' and 'schema' for text sources).",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID to use as source."},
                    "plugin": {"type": "string", "description": "Source plugin override (e.g. 'csv'). Inferred from MIME type if omitted."},
                    "on_success": {
                        "type": "string",
                        "description": (
                            "Connection-name string the source PUBLISHES. Some downstream consumer "
                            "(node 'input' or output 'sink_name') MUST equal this value. Despite the "
                            "field name, this is NOT a node id — connections match by string, not by "
                            "topology."
                        ),
                        "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
                    },
                    "on_validation_failure": {
                        "type": "string",
                        "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                        "default": _DEFAULT_SOURCE_VALIDATION_FAILURE,
                    },
                    "options": {
                        "type": "object",
                        "description": "Plugin-specific config (merged with blob path). Required fields vary by plugin: "
                        "text sources need 'column' (output field name) and 'schema' (e.g., {mode: 'observed'}).",
                    },
                },
                "required": ["blob_id", "on_success"],
            },
        },
        {
            "name": "create_blob",
            "description": "Create a new file (blob) from inline content. "
            "Use this to create seed input files (URLs, JSON, CSV snippets) "
            "mid-conversation without requiring manual upload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename for the blob (e.g. 'urls.csv', 'seed.json').",
                    },
                    "mime_type": {
                        "type": "string",
                        "enum": [
                            "text/plain",
                            "application/json",
                            "text/csv",
                            "application/x-jsonlines",
                            "application/jsonl",
                            "text/jsonl",
                        ],
                        "description": "MIME type of the content.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content as a string.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the file's purpose.",
                    },
                },
                "required": ["filename", "mime_type", "content"],
            },
        },
        {
            "name": "update_blob",
            "description": "Update the content of an existing blob (file). Overwrites the file content while preserving metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New file content.",
                    },
                },
                "required": ["blob_id", "content"],
            },
        },
        {
            "name": "delete_blob",
            "description": "Delete a blob (file) and its storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to delete.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "get_blob_content",
            "description": "Retrieve the content of a blob (file) for inspection. Large files are truncated to 50,000 characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to read.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "list_recipes",
            "description": (
                "List the registered pipeline recipes — deterministic scaffolds for common simple "
                "intents. Each recipe declares its required slots; apply_pipeline_recipe then "
                "instantiates the recipe with operator-supplied slot values. Recipes accelerate "
                "the highest-frequency 'classify CSV with LLM' and 'split rows by threshold' "
                "patterns; for shapes outside the recipe set, hand-author with set_pipeline."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "apply_pipeline_recipe",
            "description": (
                "Apply a registered pipeline recipe with operator-supplied slot values and replace "
                "the current pipeline state with the resulting configuration. Slots are validated "
                "against the recipe's declared schema before scaffolding — invalid slots are "
                "rejected with a repair hint. Call list_recipes to discover available recipes and "
                "their slot schemas. The resulting state is identical to a hand-authored "
                "set_pipeline call; the model can refine via patch_*_options afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "Recipe identifier (e.g., 'classify-rows-llm-jsonl')",
                    },
                    "slots": {
                        "type": "object",
                        "description": "Operator-supplied slot values; must match the recipe's slot schema",
                    },
                },
                "required": ["recipe_name", "slots"],
            },
        },
        {
            "name": "inspect_source",
            "description": (
                "Return bounded structural facts about a blob-backed source: source kind, observed "
                "headers, sample row count, inferred scalar types per column, URL candidates, and "
                "warnings. Reads at most 8 KiB of the blob and parses at most 100 rows. Use this "
                "before declaring a fixed CSV/JSON schema — observed headers and inferred types "
                "tell you which fields the source actually contains and what numeric coercion is "
                "needed before any gate or value_transform numeric op. Never returns raw row "
                "content; only summary facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to inspect.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        # Secret tools
        {
            "name": "list_secret_refs",
            "description": "List available secret references (API keys, credentials). Shows names and scopes, never values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "validate_secret_ref",
            "description": "Check if a secret reference exists and is accessible to the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name (e.g. 'OPENROUTER_API_KEY')."},
                },
                "required": ["name"],
            },
        },
        # Composer-LLM-callable tool surface for surfacing an interpretation
        # of a subjective or under-specified term for user review.
        # The description below is normative documentation for the LLM (mirrored
        # in the composer skill markdown) and is reviewed by the audit panel as
        # part of the request_interpretation_review event row's provenance.
        #
        # Position note: this tool is inserted BEFORE ``wire_secret_ref`` so
        # the trailing tool name remains ``wire_secret_ref`` — the Anthropic
        # cache-marker test (``test_trailing_tool_name_is_locked``) pins the
        # trailing position to preserve prompt-cache stability across deploys.
        {
            "name": "request_interpretation_review",
            "description": (
                "Ask the user to review your interpretation of a subjective or "
                "underspecified term they used. Call this BEFORE you finalise "
                "the prompt template for any LLM transform whose prompt depends "
                "on the term. Surface ONE term per call. The composition state "
                "MUST already contain the affected LLM transform (call upsert_node "
                "first) and its prompt_template MUST contain the placeholder "
                "{{interpretation:<term>}}. The user will see your draft and "
                "either accept it or amend it. Do not ask the user in assistant "
                "prose; this tool is the review surface. If no composition state "
                "exists yet, stage the LLM transform with a placeholder first, "
                "wait for that tool result, then call this tool. Do not call this "
                "for concrete operators (e.g., 'rate 1-10') or for terms the "
                "user already defined in the conversation."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["affected_node_id", "user_term", "llm_draft"],
                "properties": {
                    "affected_node_id": {
                        "type": "string",
                        "description": "node_id of the LLM transform whose prompt template depends on this term",
                    },
                    "user_term": {
                        "type": "string",
                        "description": "The user-provided term, verbatim (e.g., 'cool', 'important', 'risky')",
                    },
                    "llm_draft": {
                        "type": "string",
                        "description": "Your draft interpretation of the term, in your own words, suitable to embed as a phrase in the prompt template",
                    },
                },
            },
        },
        {
            "name": "wire_secret_ref",
            "description": "Place a secret reference marker in the pipeline config. The secret will be resolved at execution time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name."},
                    "target": {
                        "type": "string",
                        "enum": ["source", "node", "output"],
                        "description": "Which component to wire the secret into.",
                    },
                    "target_id": {"type": "string", "description": "Node ID or output name (required for node/output targets)."},
                    "option_key": {"type": "string", "description": "Config option key to set (e.g. 'api_key')."},
                },
                "required": ["name", "target", "option_key"],
            },
        },
    ]


_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_sources": _handle_list_sources,
    "list_transforms": _handle_list_transforms,
    "list_sinks": _handle_list_sinks,
    "get_plugin_schema": _handle_get_plugin_schema,
    "get_expression_grammar": _handle_get_expression_grammar,
    "explain_validation_error": _execute_explain_validation_error,
    "get_plugin_assistance": _execute_get_plugin_assistance,
    "list_models": _execute_list_models,
    "get_audit_info": _execute_get_audit_info,
    "list_recipes": _execute_list_recipes,
    "get_pipeline_state": _execute_get_pipeline_state,
    "preview_pipeline": _execute_preview_pipeline,
    "diff_pipeline": _execute_diff_pipeline,
}

_CACHEABLE_DISCOVERY_TOOLS: frozenset[str] = frozenset(_DISCOVERY_TOOLS.keys()) - {
    "diff_pipeline",
    "get_pipeline_state",
    "preview_pipeline",
}


_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source": _handle_set_source,
    "upsert_node": _handle_upsert_node,
    "upsert_edge": _handle_upsert_edge,
    "remove_node": _handle_remove_node,
    "remove_edge": _handle_remove_edge,
    "set_metadata": _handle_set_metadata,
    "set_output": _handle_set_output,
    "remove_output": _handle_remove_output,
    "patch_source_options": _handle_patch_source_options,
    "patch_node_options": _handle_patch_node_options,
    "patch_output_options": _handle_patch_output_options,
    "set_pipeline": _handle_set_pipeline,
    "clear_source": _handle_clear_source,
}

_BLOB_DISCOVERY_TOOLS: dict[str, BlobToolHandler] = {
    "list_blobs": _handle_list_blobs,
    "get_blob_metadata": _handle_get_blob_metadata,
    "get_blob_content": _execute_get_blob_content,
    "inspect_source": _execute_inspect_source,
}


_BLOB_MUTATION_TOOLS: dict[str, BlobToolHandler] = {
    "set_source_from_blob": _execute_set_source_from_blob,
    "create_blob": _execute_create_blob,
    "update_blob": _execute_update_blob,
    "delete_blob": _execute_delete_blob,
    "apply_pipeline_recipe": _execute_apply_pipeline_recipe,
}

_SECRET_DISCOVERY_TOOLS: dict[str, SecretToolHandler] = {
    "list_secret_refs": _handle_list_secret_refs,
    "validate_secret_ref": _handle_validate_secret_ref,
}


_SECRET_MUTATION_TOOLS: dict[str, SecretToolHandler] = {
    "wire_secret_ref": _execute_wire_secret_ref,
}

_all_tools = (
    set(_DISCOVERY_TOOLS)
    | set(_MUTATION_TOOLS)
    | set(_BLOB_DISCOVERY_TOOLS)
    | set(_BLOB_MUTATION_TOOLS)
    | set(_SECRET_DISCOVERY_TOOLS)
    | set(_SECRET_MUTATION_TOOLS)
)
assert len(_all_tools) == (
    len(_DISCOVERY_TOOLS)
    + len(_MUTATION_TOOLS)
    + len(_BLOB_DISCOVERY_TOOLS)
    + len(_BLOB_MUTATION_TOOLS)
    + len(_SECRET_DISCOVERY_TOOLS)
    + len(_SECRET_MUTATION_TOOLS)
), "Tool registry overlap detected"

assert set(_DISCOVERY_TOOLS) >= _CACHEABLE_DISCOVERY_TOOLS, (
    f"Cacheable tools not in discovery registry: {_CACHEABLE_DISCOVERY_TOOLS - set(_DISCOVERY_TOOLS)}"
)


def is_discovery_tool(name: str) -> bool:
    """Return True if the tool is a discovery (read-only) tool."""
    return name in _DISCOVERY_TOOLS or name in _BLOB_DISCOVERY_TOOLS or name in _SECRET_DISCOVERY_TOOLS


def is_mutation_tool(name: str) -> bool:
    """Return True when a composer tool can mutate session state or owned artifacts."""
    return name in _MUTATION_TOOLS or name in _BLOB_MUTATION_TOOLS or name in _SECRET_MUTATION_TOOLS


def is_cacheable_discovery_tool(name: str) -> bool:
    """Return True if the tool's results can be cached within a compose() call."""
    return name in _CACHEABLE_DISCOVERY_TOOLS


def _inject_prior_validation(
    result: ToolResult,
    prior: ValidationSummary,
) -> ToolResult:
    """Attach prior validation to a successful mutation result for delta computation.

    Returns the result unchanged if the mutation failed or already carries
    prior_validation (set explicitly by the handler).
    """
    if result.success and result.prior_validation is None:
        return replace(result, prior_validation=prior)
    return result


def _augment_with_plugin_schemas(
    result: ToolResult,
    tool_name: str,
    catalog: CatalogService,
) -> ToolResult:
    """Attach inline ``plugin_schemas`` to a failed option-shape rejection.

    For the mutation tools listed in
    ``_PLUGIN_SCHEMA_AUGMENTATION_TOOLS``, scan ``result.validation.errors``
    for ``Invalid options for <kind> '<plugin>'`` messages and embed the
    full ``get_plugin_schema`` payload for every named plugin. Eliminates
    the second round-trip the LLM would otherwise burn calling
    ``get_plugin_schema`` after each rejection (see composer session
    47cfbb5e on staging: 13 tool calls + 18 LLM rounds to converge a
    4-plugin pipeline because the model never preloaded schemas).

    No-op when the mutation succeeded, when no error message matches the
    option-shape pattern, when the result already carries
    ``plugin_schemas`` (handler set it directly), or when ``tool_name`` is
    not one of the augmentation-eligible tools.
    """
    if not should_augment_with_plugin_schemas(tool_name):
        return result
    if result.success or result.plugin_schemas is not None:
        return result
    schemas = build_plugin_schemas_for_failure(result, catalog)
    if schemas is None:
        return result
    return replace(result, plugin_schemas=schemas)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    secret_service: Any | None = None,
    user_id: str | None = None,
    baseline: CompositionState | None = None,
    prior_validation: ValidationSummary | None = None,
    runtime_preflight: RuntimePreflight | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
    user_message_id: str | None = None,
) -> ToolResult:
    """Execute a composition tool by name.

    Dispatches via registry dict. Discovery tools return data without
    modifying state. Mutation tools return ToolResult with updated state
    and validation. Unknown tool names return a failure result.

    Args:
        data_dir: Base data directory for S2 path allowlist enforcement.
            When provided, source options containing ``path`` or ``file``
            keys are restricted to ``{data_dir}/blobs/``.
        session_engine: SQLAlchemy engine for the session database.
            Required for blob tools to perform synchronous blob lookups.
        session_id: Current session ID. Required for blob tools.
        secret_service: WebSecretService instance. Required for secret tools.
        user_id: Current user ID. Required for secret tools.
        baseline: Baseline state for diff_pipeline comparisons.
        prior_validation: Pre-computed validation for the current state.
            When provided, mutation tools reuse this instead of calling
            state.validate() for the pre-mutation delta. Callers should
            thread the previous ToolResult.validation forward — the state
            is immutable, so validation is deterministic.
        runtime_preflight: Optional callback for runtime-equivalent preflight.
            Only applied to preview_pipeline. Pre-computed in the async
            compose loop and injected here as a cheap synchronous callback
            so execute_tool() stays synchronous.
        max_blob_storage_per_session_bytes: Configured per-session blob
            storage quota for assistant-created session artifacts. Defaults to
            the historical BlobServiceImpl-compatible value for direct tests
            and non-web callers.
    """
    # preview_pipeline has an extended signature with runtime_preflight kwarg
    # plus session context (session_engine, session_id) so the proof step
    # can inspect blob-backed sources.
    if tool_name == "preview_pipeline":
        result = _execute_preview_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            runtime_preflight=runtime_preflight,
            session_engine=session_engine,
            session_id=session_id,
        )
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    # diff_pipeline has an extended signature with baseline kwarg
    if tool_name == "diff_pipeline":
        result = _execute_diff_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            baseline=baseline,
            current_validation=prior_validation,
        )
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    # set_pipeline has the standard mutation shape for ordinary sources, but
    # can also own source.inline_blob, which requires session context to create
    # the backing blob before returning the new state.
    if tool_name == "set_pipeline":
        prior = prior_validation if prior_validation is not None else state.validate()
        result = _execute_set_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            session_engine=session_engine,
            session_id=session_id,
            user_message_id=user_message_id,
            max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
        )
        result = _inject_prior_validation(result, prior)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    # Check standard tools first
    discovery_handler = _DISCOVERY_TOOLS.get(tool_name)
    if discovery_handler is not None:
        result = discovery_handler(arguments, state, catalog, data_dir)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    mutation_handler = _MUTATION_TOOLS.get(tool_name)
    if mutation_handler is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        result = mutation_handler(arguments, state, catalog, data_dir)
        result = _inject_prior_validation(result, prior)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    # Check blob tools (extended signature with session context)
    blob_discovery = _BLOB_DISCOVERY_TOOLS.get(tool_name)
    if blob_discovery is not None:
        result = blob_discovery(arguments, state, catalog, data_dir, session_engine=session_engine, session_id=session_id)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    blob_mutation = _BLOB_MUTATION_TOOLS.get(tool_name)
    if blob_mutation is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        blob_kwargs: dict[str, Any] = {
            "session_engine": session_engine,
            "session_id": session_id,
        }
        if tool_name in _BLOB_QUOTA_MUTATION_TOOLS:
            blob_kwargs["max_blob_storage_per_session_bytes"] = max_blob_storage_per_session_bytes
        # ``create_blob`` writes the blob row with a
        # ``created_from_message_id`` provenance pointer. Only tools that
        # actually persist a new blob need the kwarg; ``set_source_from_blob``,
        # ``delete_blob``, and ``update_blob`` operate on existing rows
        # whose provenance is fixed at create time.
        if tool_name in _BLOB_PROVENANCE_MUTATION_TOOLS:
            blob_kwargs["user_message_id"] = user_message_id
        result = blob_mutation(
            arguments,
            state,
            catalog,
            data_dir,
            **blob_kwargs,
        )
        result = _inject_prior_validation(result, prior)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    # Check secret tools (extended signature with secret_service + user_id)
    secret_discovery = _SECRET_DISCOVERY_TOOLS.get(tool_name)
    if secret_discovery is not None:
        result = secret_discovery(arguments, state, catalog, data_dir, secret_service=secret_service, user_id=user_id)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    secret_mutation = _SECRET_MUTATION_TOOLS.get(tool_name)
    if secret_mutation is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        result = secret_mutation(arguments, state, catalog, data_dir, secret_service=secret_service, user_id=user_id)
        result = _inject_prior_validation(result, prior)
        return _augment_with_plugin_schemas(result, tool_name, catalog)

    return _failure_result(state, f"Unknown tool: {tool_name}")


# Module-level assertions — F-18 dual-registry invariant enforcement.
#
# These execute at module import, so a regression (e.g., copy-pasting an async
# handler into a sync registry, or registering one tool name in two registries)
# fails the build before any compose() call could trigger silent
# "coroutine was never awaited" warnings or first-registry-wins overrides.
_all_tools_v2 = (
    set(_DISCOVERY_TOOLS)
    | set(_MUTATION_TOOLS)
    | set(_BLOB_DISCOVERY_TOOLS)
    | set(_BLOB_MUTATION_TOOLS)
    | set(_SECRET_DISCOVERY_TOOLS)
    | set(_SECRET_MUTATION_TOOLS)
    | set(_SESSION_AWARE_TOOL_HANDLERS)
)
assert len(_all_tools_v2) == (
    len(_DISCOVERY_TOOLS)
    + len(_MUTATION_TOOLS)
    + len(_BLOB_DISCOVERY_TOOLS)
    + len(_BLOB_MUTATION_TOOLS)
    + len(_SECRET_DISCOVERY_TOOLS)
    + len(_SECRET_MUTATION_TOOLS)
    + len(_SESSION_AWARE_TOOL_HANDLERS)
), (
    "Tool registry overlap detected — a tool name appears in more than one of "
    "_DISCOVERY_TOOLS / _MUTATION_TOOLS / blob / secret / _SESSION_AWARE_TOOL_HANDLERS"
)

# Every session-aware handler must be a coroutine function. A sync function
# accidentally registered here would silently return a non-Awaitable; the
# compose-loop ``await`` would crash with TypeError at the worst time.
for _name, _handler in _SESSION_AWARE_TOOL_HANDLERS.items():
    assert asyncio.iscoroutinefunction(_handler), (
        f"_SESSION_AWARE_TOOL_HANDLERS[{_name!r}] is not async; sync handlers belong in _MUTATION_TOOLS / _DISCOVERY_TOOLS instead."
    )

# Every sync-registry handler must NOT be a coroutine. Catches the reverse
# regression: an async handler dropped into the sync dispatch path that
# would return a coroutine object as if it were a ToolResult.
#
# The six sync registries have heterogeneous handler value-types (the blob
# and secret registries carry handlers with extra session-context kwargs),
# so the local ``_sync_registry`` is typed broadly as
# ``Mapping[str, Callable[..., Any]]`` for the duration of this check.
_sync_registries_for_check: tuple[tuple[str, Mapping[str, Callable[..., Any]]], ...] = (
    ("_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _DISCOVERY_TOOLS)),
    ("_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _MUTATION_TOOLS)),
    ("_BLOB_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_DISCOVERY_TOOLS)),
    ("_BLOB_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_MUTATION_TOOLS)),
    ("_SECRET_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_DISCOVERY_TOOLS)),
    ("_SECRET_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_MUTATION_TOOLS)),
)
for _sync_registry_name, _sync_registry in _sync_registries_for_check:
    for _name, _handler in _sync_registry.items():
        assert not asyncio.iscoroutinefunction(_handler), (
            f"{_sync_registry_name}[{_name!r}] is async; async handlers belong in _SESSION_AWARE_TOOL_HANDLERS instead."
        )
