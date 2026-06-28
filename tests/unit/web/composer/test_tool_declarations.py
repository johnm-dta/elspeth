"""Tests for the ToolDeclaration primitive and the create_blob migration.

Ticket: elspeth-6c9972ccbf (Composer tools — ToolDeclaration paradigm).

Step 1 introduces ``ToolDeclaration`` and migrates ``create_blob`` as the
exemplar. The migration must be byte-identity-preserving — the schema the LLM
sees pre- and post-migration is identical — and the declaration's invariants
must crash early when violated (offensive-programming policy, see CLAUDE.md).

These tests guard the migration's correctness independent of the import-time
parity assertions in ``_dispatch.py``: parity asserts the declaration agrees
with the registry as it stands today; these tests pin the post-migration
shape (so a future refactor cannot silently re-author the LLM-facing schema).
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.tools._dispatch import get_tool_definitions
from elspeth.web.composer.tools.blobs import (
    _CREATE_BLOB_DECLARATION,
    _DELETE_BLOB_DECLARATION,
    _LIST_COMPOSER_BLOBS_DECLARATION,
    _UPDATE_BLOB_DECLARATION,
    _WIRE_BLOB_INLINE_REF_DECLARATION,
    _execute_create_blob,
    _execute_delete_blob,
    _execute_update_blob,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
    assert_unique_names,
    derive_blob_store_only_names,
    derive_cacheable_names,
    derive_handler_map_for,
    derive_name_set_for,
    derive_tool_definitions_by_name,
)
from elspeth.web.composer.tools.sessions import (
    _APPLY_PIPELINE_RECIPE_DECLARATION,
    _execute_apply_pipeline_recipe,
)
from elspeth.web.composer.tools.sources import (
    _SET_SOURCE_FROM_BLOB_DECLARATION,
    _execute_set_source_from_blob,
)

_EXPECTED_CREATE_BLOB_DEFINITION: dict[str, object] = {
    "name": "create_blob",
    "description": (
        "Create a new file (blob) from inline content. "
        "Use this to create seed input files (URLs, JSON, CSV snippets) "
        "mid-conversation without requiring manual upload."
    ),
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
        "additionalProperties": False,
    },
}


_EXPECTED_UPDATE_BLOB_DEFINITION: dict[str, object] = {
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
        "additionalProperties": False,
    },
}


_EXPECTED_DELETE_BLOB_DEFINITION: dict[str, object] = {
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
        "additionalProperties": False,
    },
}


_EXPECTED_APPLY_PIPELINE_RECIPE_DEFINITION: dict[str, object] = {
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
        "additionalProperties": False,
    },
}


class TestCreateBlobMigration:
    """The migration must be byte-identity-preserving for create_blob."""

    def test_get_tool_definitions_emits_expected_create_blob_definition(self) -> None:
        """The LLM-facing schema for create_blob is exactly the pre-migration shape."""
        definitions = get_tool_definitions()
        create_blob = next(d for d in definitions if d["name"] == "create_blob")
        assert create_blob == _EXPECTED_CREATE_BLOB_DEFINITION

    def test_declaration_handler_is_execute_create_blob(self) -> None:
        """The declaration's handler is the same callable the registry holds."""
        assert _CREATE_BLOB_DECLARATION.handler is _execute_create_blob

    def test_declaration_kind_is_blob_mutation(self) -> None:
        assert _CREATE_BLOB_DECLARATION.kind is ToolKind.BLOB_MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """create_blob is blob-store-only (never advances CompositionState)."""
        assert _CREATE_BLOB_DECLARATION.blob_store_only is True

    def test_declaration_is_not_cacheable(self) -> None:
        """Mutations are forbidden from being cacheable; create_blob is no exception."""
        assert _CREATE_BLOB_DECLARATION.cacheable is False


class TestUpdateBlobMigration:
    """update_blob migration: byte-identity + correct kind (BLOB_MUTATION, store-only)."""

    def test_get_tool_definitions_emits_expected_update_blob_definition(self) -> None:
        definitions = get_tool_definitions()
        update_blob = next(d for d in definitions if d["name"] == "update_blob")
        assert update_blob == _EXPECTED_UPDATE_BLOB_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _UPDATE_BLOB_DECLARATION.handler is _execute_update_blob

    def test_declaration_kind_is_blob_mutation(self) -> None:
        assert _UPDATE_BLOB_DECLARATION.kind is ToolKind.BLOB_MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """update_blob is blob-store-only (never advances CompositionState)."""
        assert _UPDATE_BLOB_DECLARATION.blob_store_only is True


class TestDeleteBlobMigration:
    """delete_blob migration: byte-identity + correct kind (BLOB_MUTATION, store-only)."""

    def test_get_tool_definitions_emits_expected_delete_blob_definition(self) -> None:
        definitions = get_tool_definitions()
        delete_blob = next(d for d in definitions if d["name"] == "delete_blob")
        assert delete_blob == _EXPECTED_DELETE_BLOB_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _DELETE_BLOB_DECLARATION.handler is _execute_delete_blob

    def test_declaration_blob_kwarg_shape(self) -> None:
        """delete_blob removes a row; blob-store-only (never advances CompositionState)."""
        assert _DELETE_BLOB_DECLARATION.blob_store_only is True


class TestSetSourceFromBlobMigration:
    """set_source_from_blob migration: byte-identity + correct kwarg shape (no blob-store-only)."""

    def test_get_tool_definitions_emits_expected_set_source_from_blob_definition(self) -> None:
        from elspeth.web.composer.tools._common import (
            _DEFAULT_SOURCE_VALIDATION_FAILURE,
            _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
        )

        expected = {
            "name": "set_source_from_blob",
            "description": (
                "Wire a blob as the pipeline source. Resolves the blob's storage path "
                "internally and infers the source plugin from its MIME type. "
                "Use 'options' for plugin-specific config (e.g., 'column' and 'schema' for text sources)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID to use as source."},
                    "source_name": {
                        "type": "string",
                        "description": "Source root name to bind. Defaults to 'source' for legacy single-source pipelines.",
                    },
                    "plugin": {
                        "type": "string",
                        "description": "Source plugin override (e.g. 'csv'). Inferred from MIME type if omitted.",
                    },
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
                        "description": (
                            "Plugin-specific config (merged with blob path). Required fields vary by plugin: "
                            "text sources need 'column' (output field name) and 'schema' (e.g., {mode: 'observed'})."
                        ),
                    },
                },
                "required": ["blob_id", "on_success"],
                "additionalProperties": False,
            },
        }
        definitions = get_tool_definitions()
        emitted = next(d for d in definitions if d["name"] == "set_source_from_blob")
        assert emitted == expected

    def test_declaration_handler_matches(self) -> None:
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.handler is _execute_set_source_from_blob

    def test_declaration_kind_is_blob_mutation(self) -> None:
        """Despite changing CompositionState, set_source_from_blob is dispatched via the blob path
        (its handler reads session_engine + session_id from ToolContext to resolve the blob row)."""
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.kind is ToolKind.BLOB_MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """set_source_from_blob does not WRITE a blob — not store-only (it advances
        CompositionState by setting the source)."""
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.blob_store_only is False


class TestApplyPipelineRecipeMigration:
    """apply_pipeline_recipe migration: byte-identity + MUTATION kind (advances CompositionState;
    does not create blobs on its own dispatch path — recipe slots cannot carry inline_blob)."""

    def test_get_tool_definitions_emits_expected_apply_pipeline_recipe_definition(self) -> None:
        definitions = get_tool_definitions()
        emitted = next(d for d in definitions if d["name"] == "apply_pipeline_recipe")
        assert emitted == _EXPECTED_APPLY_PIPELINE_RECIPE_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.handler is _execute_apply_pipeline_recipe

    def test_declaration_kind_is_mutation(self) -> None:
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.kind is ToolKind.MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """apply_pipeline_recipe is not blob-store-only — it replaces CompositionState."""
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.blob_store_only is False


class TestStep2RegistryAggregation:
    """The blob-mutation declarations are aggregated into _REGISTERED_TOOLS at import time."""

    def test_all_blob_mutation_tools_are_declared(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        declared = {d.name for d in _REGISTERED_TOOLS if d.kind is ToolKind.BLOB_MUTATION}
        expected = {
            "create_blob",
            "update_blob",
            "delete_blob",
            "set_source_from_blob",
            "wire_blob_inline_ref",
        }
        assert declared == expected

    def test_registered_tools_count_at_least_five(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        # Step 2 registered blob-mutation tools plus apply_pipeline_recipe
        # (MUTATION kind). Step 3 adds more tiers (discovery first); the count
        # strictly grows as tiers migrate.
        assert len(_REGISTERED_TOOLS) >= 5

    def test_request_interpretation_review_is_not_a_normal_tool_declaration(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        assert "request_interpretation_review" not in {d.name for d in _REGISTERED_TOOLS}


class TestStep3DiscoveryTierMigration:
    """All 13 discovery tools must carry declarations with byte-identical schemas.

    Each tool's ``json_schema`` field on its ``ToolDeclaration`` must round-trip
    through ``derive_tool_definitions_by_name`` to the same JSON shape the
    LLM saw before the migration. The fixed-expected-dict comparisons below
    pin that shape so a future drift cannot silently re-author the LLM-facing
    schema.
    """

    def _get(self, name: str) -> dict[str, object]:
        return next(d for d in get_tool_definitions() if d["name"] == name)

    def test_list_sources(self) -> None:
        assert self._get("list_sources") == {
            "name": "list_sources",
            "description": "List available source plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_list_transforms(self) -> None:
        assert self._get("list_transforms") == {
            "name": "list_transforms",
            "description": "List available transform plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_list_sinks(self) -> None:
        assert self._get("list_sinks") == {
            "name": "list_sinks",
            "description": "List available sink plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_get_plugin_schema(self) -> None:
        assert self._get("get_plugin_schema") == {
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
                "additionalProperties": False,
            },
        }

    def test_get_expression_grammar(self) -> None:
        assert self._get("get_expression_grammar") == {
            "name": "get_expression_grammar",
            "description": "Get the gate expression syntax reference.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_explain_validation_error(self) -> None:
        assert self._get("explain_validation_error") == {
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
                "additionalProperties": False,
            },
        }

    def test_get_plugin_assistance(self) -> None:
        assert self._get("get_plugin_assistance") == {
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
                "additionalProperties": False,
            },
        }

    def test_list_models(self) -> None:
        assert self._get("list_models") == {
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
                "additionalProperties": False,
            },
        }

    def test_get_audit_info(self) -> None:
        assert self._get("get_audit_info") == {
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
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_list_recipes(self) -> None:
        assert self._get("list_recipes") == {
            "name": "list_recipes",
            "description": (
                "List the registered pipeline recipes — deterministic scaffolds for common simple "
                "intents. Each recipe declares its required slots; apply_pipeline_recipe then "
                "instantiates the recipe with operator-supplied slot values. Recipes accelerate "
                "the highest-frequency 'classify CSV with LLM' and 'split rows by threshold' "
                "patterns; for shapes outside the recipe set, hand-author with set_pipeline."
            ),
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_get_pipeline_state(self) -> None:
        assert self._get("get_pipeline_state") == {
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
                "additionalProperties": False,
            },
        }

    def test_preview_pipeline(self) -> None:
        assert self._get("preview_pipeline") == {
            "name": "preview_pipeline",
            "description": "Preview the current pipeline configuration — returns "
            "validation status, source summary, and node/output overview "
            "without executing. Use this to confirm the pipeline is set up "
            "correctly before running.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_diff_pipeline(self) -> None:
        assert self._get("diff_pipeline") == {
            "name": "diff_pipeline",
            "description": "Show what changed since the session was loaded or created. "
            "Returns added, removed, and modified nodes/edges/outputs, "
            "plus warnings introduced or resolved.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_cacheable_subset_is_correct(self) -> None:
        """The 10 discovery tools that should be cacheable are; the 3
        session-mutable ones are not."""
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS
        from elspeth.web.composer.tools.declarations import derive_cacheable_names

        cacheable = derive_cacheable_names(_REGISTERED_TOOLS)
        expected_cacheable = {
            "list_sources",
            "list_transforms",
            "list_sinks",
            "get_plugin_schema",
            "get_expression_grammar",
            "explain_validation_error",
            "get_plugin_assistance",
            "list_models",
            "get_audit_info",
            "list_recipes",
        }
        assert expected_cacheable <= cacheable
        # The three session-mutable discovery tools MUST NOT be cacheable.
        assert "get_pipeline_state" not in cacheable
        assert "preview_pipeline" not in cacheable
        assert "diff_pipeline" not in cacheable


class TestStep3MutationTierMigration:
    """All 13 standard mutation tools must carry declarations with byte-identical schemas.

    These tests pin the post-migration JSON shape so a future drift cannot silently
    re-author the LLM-facing schema. The schemas for ``upsert_node`` and ``set_pipeline``
    are particularly load-bearing — they document the connection-name vocabulary the
    LLM uses to wire pipelines, and any drift would break composer convergence.
    """

    def _get(self, name: str) -> dict[str, object]:
        return next(d for d in get_tool_definitions() if d["name"] == name)

    def test_set_source_kind_and_handler(self) -> None:
        from elspeth.web.composer.tools.sources import _SET_SOURCE_DECLARATION, _handle_set_source

        assert _SET_SOURCE_DECLARATION.kind is ToolKind.MUTATION
        assert _SET_SOURCE_DECLARATION.handler is _handle_set_source
        assert _SET_SOURCE_DECLARATION.cacheable is False

    def test_clear_source(self) -> None:
        assert self._get("clear_source") == {
            "name": "clear_source",
            "description": "Remove a named source from the pipeline composition state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string", "description": "Source root name to clear. Defaults to 'source'."},
                },
                "required": [],
                "additionalProperties": False,
            },
        }

    def test_remove_node(self) -> None:
        assert self._get("remove_node") == {
            "name": "remove_node",
            "description": "Remove a node and all its edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove."},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        }

    def test_remove_edge(self) -> None:
        assert self._get("remove_edge") == {
            "name": "remove_edge",
            "description": "Remove an edge by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Edge ID to remove."},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        }

    def test_remove_output(self) -> None:
        assert self._get("remove_output") == {
            "name": "remove_output",
            "description": "Remove a pipeline output (sink) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {"type": "string", "description": "Sink name to remove."},
                },
                "required": ["sink_name"],
                "additionalProperties": False,
            },
        }

    def test_set_metadata(self) -> None:
        assert self._get("set_metadata") == {
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
                "additionalProperties": False,
            },
        }

    def test_patch_source_options(self) -> None:
        assert self._get("patch_source_options") == {
            "name": "patch_source_options",
            "description": "Apply a shallow merge-patch to a named source's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_name": {"type": "string", "description": "Source root name to patch. Defaults to 'source'."},
                    "patch": {
                        "type": "object",
                        "description": "Merge-patch to apply to source options.",
                    },
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        }

    def test_set_source_description_byte_identity(self) -> None:
        """The set_source description references _SOURCE_VALIDATION_FAILURE_DESCRIPTION;
        verify the resolved value lands byte-identical in the emitted def."""
        from elspeth.web.composer.tools._common import _SOURCE_VALIDATION_FAILURE_DESCRIPTION

        defn = self._get("set_source")
        params = defn["parameters"]
        assert isinstance(params, dict)
        properties = params["properties"]
        assert isinstance(properties, dict)
        on_vf = properties["on_validation_failure"]
        assert isinstance(on_vf, dict)
        assert on_vf["description"] == _SOURCE_VALIDATION_FAILURE_DESCRIPTION

    def test_set_pipeline_source_validation_failure_resolved(self) -> None:
        """set_pipeline carries the same constant; same drift check."""
        from elspeth.web.composer.tools._common import _SOURCE_VALIDATION_FAILURE_DESCRIPTION

        defn = self._get("set_pipeline")
        params = defn["parameters"]
        assert isinstance(params, dict)
        source = params["properties"]["source"]
        assert isinstance(source, dict)
        on_vf = source["properties"]["on_validation_failure"]
        assert isinstance(on_vf, dict)
        assert on_vf["description"] == _SOURCE_VALIDATION_FAILURE_DESCRIPTION

    def test_set_pipeline_required_top_level(self) -> None:
        """set_pipeline.required is exactly ['nodes', 'edges', 'outputs'].

        Multi-source: a caller may supply either the singular ``source`` or the
        ``sources`` map (both are optional properties), so neither is required.
        """
        defn = self._get("set_pipeline")
        params = defn["parameters"]
        assert isinstance(params, dict)
        assert params["required"] == ["nodes", "edges", "outputs"]

    def test_upsert_node_required(self) -> None:
        defn = self._get("upsert_node")
        params = defn["parameters"]
        assert isinstance(params, dict)
        assert params["required"] == ["id", "node_type", "input"]

    def test_upsert_edge_required(self) -> None:
        defn = self._get("upsert_edge")
        params = defn["parameters"]
        assert isinstance(params, dict)
        assert params["required"] == ["id", "from_node", "to_node", "edge_type"]

    def test_no_mutation_tool_is_cacheable(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        mutations = [d for d in _REGISTERED_TOOLS if d.kind is ToolKind.MUTATION]
        for d in mutations:
            assert d.cacheable is False, f"{d.name} mutation must not be cacheable"
        # Confirm we have all 14 standard mutations.
        names = {d.name for d in mutations}
        expected = {
            "set_source",
            "upsert_node",
            "upsert_edge",
            "remove_node",
            "remove_edge",
            "set_metadata",
            "set_output",
            "remove_output",
            "patch_source_options",
            "patch_node_options",
            "patch_output_options",
            "set_pipeline",
            "clear_source",
            "apply_pipeline_recipe",
        }
        assert names == expected


class TestStep3BlobDiscoveryTierMigration:
    """All blob-discovery tools must carry declarations with byte-identical schemas."""

    def _get(self, name: str) -> dict[str, object]:
        return next(d for d in get_tool_definitions() if d["name"] == name)

    def test_list_blobs(self) -> None:
        assert self._get("list_blobs") == {
            "name": "list_blobs",
            "description": "List uploaded/created files (blobs) in this session with metadata.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_list_composer_blobs(self) -> None:
        assert self._get("list_composer_blobs") == {
            "name": "list_composer_blobs",
            "description": (
                "List ready blobs available for audited inline-content authoring. "
                "Returns only blob_id, mime_type, size_bytes, content_hash, and filename; never content bytes."
            ),
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_list_composer_blobs_declaration_kind(self) -> None:
        assert _LIST_COMPOSER_BLOBS_DECLARATION.kind is ToolKind.BLOB_DISCOVERY

    def test_get_blob_metadata(self) -> None:
        assert self._get("get_blob_metadata") == {
            "name": "get_blob_metadata",
            "description": "Get metadata for a specific blob (file) by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID."},
                },
                "required": ["blob_id"],
                "additionalProperties": False,
            },
        }

    def test_get_blob_content(self) -> None:
        assert self._get("get_blob_content") == {
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
                "additionalProperties": False,
            },
        }

    def test_inspect_source(self) -> None:
        assert self._get("inspect_source") == {
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
                "additionalProperties": False,
            },
        }

    def test_all_blob_discovery_tools_registered(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        names = {d.name for d in _REGISTERED_TOOLS if d.kind is ToolKind.BLOB_DISCOVERY}
        assert names == {"list_blobs", "list_composer_blobs", "get_blob_metadata", "get_blob_content", "inspect_source"}

    def test_wire_blob_inline_ref(self) -> None:
        assert self._get("wire_blob_inline_ref") == {
            "name": "wire_blob_inline_ref",
            "description": (
                "Author a widened blob_ref inline_content marker at a canonical field_path. "
                "Composer pins sha256 from blob metadata; callers must not pass content bytes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_path": {
                        "type": "string",
                        "description": (
                            "Canonical path: source.options.<field>, source:<name>.options.<field>, "
                            "node:<node_id>.options.<field>, or output:<name>.options.<field>."
                        ),
                    },
                    "blob_id": {
                        "type": "string",
                        "format": "uuid",
                        "description": "Ready blob ID to wire as inline content.",
                    },
                    "encoding": {
                        "type": "string",
                        "enum": ["latin-1", "utf-16", "utf-8", "utf-8-sig"],
                        "default": "utf-8",
                        "description": "Text decoder used at runtime. Defaults to utf-8.",
                    },
                },
                "required": ["field_path", "blob_id"],
                "additionalProperties": False,
            },
        }

    def test_wire_blob_inline_ref_declaration_kind(self) -> None:
        assert _WIRE_BLOB_INLINE_REF_DECLARATION.kind is ToolKind.BLOB_MUTATION
        assert _WIRE_BLOB_INLINE_REF_DECLARATION.blob_store_only is False


class TestStep3SecretTierMigration:
    """All 3 secret tools must carry declarations (2 discovery + 1 mutation)."""

    def _get(self, name: str) -> dict[str, object]:
        return next(d for d in get_tool_definitions() if d["name"] == name)

    def test_list_secret_refs(self) -> None:
        assert self._get("list_secret_refs") == {
            "name": "list_secret_refs",
            "description": "List available secret references (API keys, credentials). Shows names and scopes, never values.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }

    def test_validate_secret_ref(self) -> None:
        assert self._get("validate_secret_ref") == {
            "name": "validate_secret_ref",
            "description": "Check if a secret reference exists and is accessible to the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name (e.g. 'OPENROUTER_API_KEY')."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }

    def test_wire_secret_ref(self) -> None:
        assert self._get("wire_secret_ref") == {
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
                "additionalProperties": False,
            },
        }

    def test_secret_tier_kinds(self) -> None:
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        discovery = {d.name for d in _REGISTERED_TOOLS if d.kind is ToolKind.SECRET_DISCOVERY}
        mutation = {d.name for d in _REGISTERED_TOOLS if d.kind is ToolKind.SECRET_MUTATION}
        assert discovery == {"list_secret_refs", "validate_secret_ref"}
        assert mutation == {"wire_secret_ref"}

    def test_no_secret_tool_is_cacheable(self) -> None:
        """Secret state mutates outside the composer; never cache."""
        from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS

        for d in _REGISTERED_TOOLS:
            if d.kind in (ToolKind.SECRET_DISCOVERY, ToolKind.SECRET_MUTATION):
                assert d.cacheable is False, f"{d.name} secret tool must not be cacheable"

    def test_wire_secret_ref_is_last_in_get_tool_definitions(self) -> None:
        """Cache-marker stability: wire_secret_ref must remain the trailing tool name
        in get_tool_definitions() so the Anthropic prompt-cache marker stays stable
        across deploys (see _dispatch.py line ~1001 for the rationale)."""
        defs = get_tool_definitions()
        assert defs[-1]["name"] == "wire_secret_ref"


class TestToolDeclarationInvariants:
    """The constructor must crash early on inconsistent declarations."""

    @staticmethod
    def _make(**overrides: object) -> ToolDeclaration:
        defaults: dict[str, object] = {
            "name": "test_tool",
            "handler": _execute_create_blob,  # any callable with the right signature
            "kind": ToolKind.DISCOVERY,
            "description": "A test tool.",
            "json_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        }
        defaults.update(overrides)
        return ToolDeclaration(**defaults)  # type: ignore[arg-type]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            self._make(name="")

    def test_non_toolkind_kind_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a ToolKind member"):
            self._make(kind="discovery")

    def test_cacheable_mutation_raises(self) -> None:
        with pytest.raises(ValueError, match="cacheable=True is forbidden"):
            self._make(kind=ToolKind.MUTATION, cacheable=True)

    def test_cacheable_blob_mutation_raises(self) -> None:
        """Inverted cacheability invariant (Python-engineer M3, 2026-05-23):
        the check fires for every non-DISCOVERY kind, not only the original
        mutation triplet."""
        with pytest.raises(ValueError, match="Only DISCOVERY tools may be cacheable"):
            self._make(kind=ToolKind.BLOB_MUTATION, cacheable=True)

    def test_cacheable_secret_mutation_raises(self) -> None:
        """Inverted cacheability invariant: SECRET_MUTATION is non-DISCOVERY
        and must reject cacheable=True at the declaration site."""
        with pytest.raises(ValueError, match="Only DISCOVERY tools may be cacheable"):
            self._make(kind=ToolKind.SECRET_MUTATION, cacheable=True)

    def test_cacheable_blob_discovery_raises(self) -> None:
        """Inverted cacheability invariant: BLOB_DISCOVERY tools are NOT
        part of the per-call cache contract (``_registry.py`` enforces
        ``_CACHEABLE_DISCOVERY_TOOL_NAMES ⊆ _DISCOVERY_TOOL_NAMES``). The
        constructor must reject ``cacheable=True`` at declaration time so
        the error surfaces at the call site of the misconfigured
        declaration, not at registry import (Python-engineer M3 review
        finding, 2026-05-23)."""
        with pytest.raises(ValueError, match="Only DISCOVERY tools may be cacheable"):
            self._make(kind=ToolKind.BLOB_DISCOVERY, cacheable=True)

    def test_cacheable_secret_discovery_raises(self) -> None:
        """Inverted cacheability invariant: SECRET_DISCOVERY tools are
        excluded from caching by the same registry subset assertion as
        BLOB_DISCOVERY. The constructor catches the misconfiguration at
        declaration time."""
        with pytest.raises(ValueError, match="Only DISCOVERY tools may be cacheable"):
            self._make(kind=ToolKind.SECRET_DISCOVERY, cacheable=True)

    def test_invalid_json_schema_raises(self) -> None:
        """Systems-thinker recommendation #3 (2026-05-23): the ``json_schema``
        field must meta-validate against Draft 2020-12 at construction time.
        Without this a malformed schema (typo in ``type``, structural error)
        escapes to the LLM API edge and fails as an opaque upstream 400."""
        with pytest.raises(ValueError, match="is not a valid JSON Schema"):
            # ``type: "objet"`` (sic) is a typo the metaschema rejects via
            # its ``simpleTypes`` enum. This catches the canonical class of
            # mistake the check defends against.
            self._make(json_schema={"type": "objet", "properties": {}, "required": []})

    def test_invalid_json_schema_structural_error_raises(self) -> None:
        """``properties`` MUST be an object per Draft 2020-12; passing a
        list instead is a structural error the metaschema catches."""
        with pytest.raises(ValueError, match="is not a valid JSON Schema"):
            self._make(json_schema={"type": "object", "properties": ["wrong"], "required": []})

    def test_json_schema_root_must_be_object(self) -> None:
        with pytest.raises(ValueError, match="root schema must be an object"):
            self._make(json_schema={"type": "array", "items": {"type": "string"}})

    def test_json_schema_root_must_be_closed(self) -> None:
        with pytest.raises(ValueError, match="additionalProperties=false"):
            self._make(json_schema={"type": "object", "properties": {}, "required": []})

    def test_json_schema_root_rejects_dynamic_additional_properties(self) -> None:
        with pytest.raises(ValueError, match="additionalProperties=false"):
            self._make(
                json_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": {"type": "string"},
                }
            )

    def test_tool_kind_has_no_session_aware_value(self) -> None:
        """SESSION_AWARE was a dead enum value advertising a shape no
        declaration carried — removed by the 2026-05-23 four-agent review
        cleanup. When ``elspeth-f5da936747`` widens ``ToolDeclaration`` to
        admit async handlers, ``SESSION_AWARE`` will be re-added together
        with the first declaration that uses it. This test pins that
        invariant so the kind cannot silently re-appear unaccompanied.
        """
        assert "SESSION_AWARE" not in ToolKind.__members__

    def test_non_blob_mutation_with_blob_store_only_raises(self) -> None:
        with pytest.raises(ValueError, match="blob_store_only=True is "):
            self._make(kind=ToolKind.SECRET_MUTATION, blob_store_only=True)

    def test_json_schema_is_deeply_frozen(self) -> None:
        """Storing the json_schema deep-freezes — mutation of source dict cannot bleed in."""
        source: dict[str, object] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        decl = self._make(json_schema=source)
        # Mutate source post-construction; the declaration must be unaffected.
        source["required"] = ["y"]
        # deep_freeze converts the inner list to a tuple.
        assert decl.json_schema["required"] == ("x",)

    def test_derivation_freezes_so_registry_cannot_be_aliased(self) -> None:
        """derive_tool_definitions_by_name returns a deeply-immutable mapping.

        Python-engineer H1 review finding (2026-05-23): the registry must be
        deeply frozen so a caller of get_tool_definitions cannot mutate the
        source-of-truth ``parameters`` dict through the returned reference.
        Emission-side thawing (in ``get_tool_definitions``) hands callers
        fresh isolated mutable copies — see
        ``test_get_tool_definitions_returns_isolated_mutable_copies`` below.
        """
        from types import MappingProxyType

        from elspeth.web.composer.tools.declarations import derive_tool_definitions_by_name

        decl = self._make(
            json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        )
        emitted = derive_tool_definitions_by_name([decl])[decl.name]
        # Outer entry is frozen.
        assert isinstance(emitted, MappingProxyType)
        # Parameters subtree is frozen — caller cannot append to ``required``.
        assert isinstance(emitted["parameters"], MappingProxyType)
        assert emitted["parameters"]["required"] == ("x",)  # tuple, not list

    def test_get_tool_definitions_returns_isolated_mutable_copies(self) -> None:
        """get_tool_definitions hands callers fresh mutable copies on every call.

        Python-engineer H1 review finding (2026-05-23): two consecutive calls
        must not share mutable state. Mutating one returned ``parameters`` dict
        must not be visible in a later call, and must not corrupt the registry.
        """
        from elspeth.web.composer.tools._dispatch import get_tool_definitions

        first = get_tool_definitions()
        second = get_tool_definitions()
        # External-facing dicts are mutable for LiteLLM / MCP compatibility.
        assert isinstance(first[0]["parameters"], dict)
        first[0]["parameters"]["required"] = ["sabotaged"]
        # Second call is not contaminated by the first call's mutation.
        assert second[0]["parameters"]["required"] != ["sabotaged"]
        # Third call also pristine — confirms the registry itself is intact.
        third = get_tool_definitions()
        assert third[0]["parameters"]["required"] != ["sabotaged"]


class TestDerivationHelpers:
    """Each derive_* helper projects an iterable of declarations purely."""

    def test_handler_map_filters_by_kind(self) -> None:
        result = derive_handler_map_for([_CREATE_BLOB_DECLARATION], ToolKind.BLOB_MUTATION)
        assert dict(result) == {"create_blob": _execute_create_blob}

    def test_handler_map_excludes_other_kinds(self) -> None:
        result = derive_handler_map_for([_CREATE_BLOB_DECLARATION], ToolKind.DISCOVERY)
        assert dict(result) == {}

    def test_name_set_for_blob_mutation(self) -> None:
        assert derive_name_set_for([_CREATE_BLOB_DECLARATION], ToolKind.BLOB_MUTATION) == {"create_blob"}

    def test_tool_definitions_by_name_round_trip(self) -> None:
        """The frozen derivation result is value-equal to the JSON-shaped
        expected fixture once deep_thaw is applied — i.e. the migration is
        round-trip safe through the freeze/thaw boundary."""
        from elspeth.contracts.freeze import deep_thaw

        result = derive_tool_definitions_by_name([_CREATE_BLOB_DECLARATION])
        assert deep_thaw(result["create_blob"]) == _EXPECTED_CREATE_BLOB_DEFINITION

    def test_blob_store_only_names_includes_create_blob(self) -> None:
        assert derive_blob_store_only_names([_CREATE_BLOB_DECLARATION]) == {"create_blob"}

    def test_cacheable_names_empty_for_blob_mutation(self) -> None:
        assert derive_cacheable_names([_CREATE_BLOB_DECLARATION]) == frozenset()


class TestAssertUniqueNames:
    """Duplicate tool names must fail at aggregation time, not at dispatch."""

    def test_unique_names_passes(self) -> None:
        assert_unique_names([_CREATE_BLOB_DECLARATION])

    def test_duplicate_names_raise(self) -> None:
        with pytest.raises(RuntimeError, match="registered more than once"):
            assert_unique_names([_CREATE_BLOB_DECLARATION, _CREATE_BLOB_DECLARATION])
