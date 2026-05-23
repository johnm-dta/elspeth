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
    _UPDATE_BLOB_DECLARATION,
    _execute_create_blob,
    _execute_delete_blob,
    _execute_update_blob,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
    assert_unique_names,
    derive_blob_provenance_names,
    derive_blob_quota_names,
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
        """create_blob needs quota, provenance, and is blob-store-only."""
        assert _CREATE_BLOB_DECLARATION.needs_blob_quota is True
        assert _CREATE_BLOB_DECLARATION.needs_blob_provenance is True
        assert _CREATE_BLOB_DECLARATION.blob_store_only is True

    def test_declaration_is_not_cacheable(self) -> None:
        """Mutations are forbidden from being cacheable; create_blob is no exception."""
        assert _CREATE_BLOB_DECLARATION.cacheable is False


class TestUpdateBlobMigration:
    """update_blob migration: byte-identity + correct kwarg shape (quota, no provenance, store-only)."""

    def test_get_tool_definitions_emits_expected_update_blob_definition(self) -> None:
        definitions = get_tool_definitions()
        update_blob = next(d for d in definitions if d["name"] == "update_blob")
        assert update_blob == _EXPECTED_UPDATE_BLOB_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _UPDATE_BLOB_DECLARATION.handler is _execute_update_blob

    def test_declaration_kind_is_blob_mutation(self) -> None:
        assert _UPDATE_BLOB_DECLARATION.kind is ToolKind.BLOB_MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """update_blob consumes quota but NOT provenance, and is blob-store-only."""
        assert _UPDATE_BLOB_DECLARATION.needs_blob_quota is True
        assert _UPDATE_BLOB_DECLARATION.needs_blob_provenance is False
        assert _UPDATE_BLOB_DECLARATION.blob_store_only is True


class TestDeleteBlobMigration:
    """delete_blob migration: byte-identity + correct kwarg shape (no quota, no provenance, store-only)."""

    def test_get_tool_definitions_emits_expected_delete_blob_definition(self) -> None:
        definitions = get_tool_definitions()
        delete_blob = next(d for d in definitions if d["name"] == "delete_blob")
        assert delete_blob == _EXPECTED_DELETE_BLOB_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _DELETE_BLOB_DECLARATION.handler is _execute_delete_blob

    def test_declaration_blob_kwarg_shape(self) -> None:
        """delete_blob removes a row; no quota or provenance kwargs, but blob-store-only."""
        assert _DELETE_BLOB_DECLARATION.needs_blob_quota is False
        assert _DELETE_BLOB_DECLARATION.needs_blob_provenance is False
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
        """set_source_from_blob does not WRITE a blob — no quota/provenance; not store-only (it
        advances CompositionState by setting the source)."""
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.needs_blob_quota is False
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.needs_blob_provenance is False
        assert _SET_SOURCE_FROM_BLOB_DECLARATION.blob_store_only is False


class TestApplyPipelineRecipeMigration:
    """apply_pipeline_recipe migration: byte-identity + correct kwarg shape (quota+provenance,
    NOT blob-store-only because it replaces CompositionState)."""

    def test_get_tool_definitions_emits_expected_apply_pipeline_recipe_definition(self) -> None:
        definitions = get_tool_definitions()
        emitted = next(d for d in definitions if d["name"] == "apply_pipeline_recipe")
        assert emitted == _EXPECTED_APPLY_PIPELINE_RECIPE_DEFINITION

    def test_declaration_handler_matches(self) -> None:
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.handler is _execute_apply_pipeline_recipe

    def test_declaration_kind_is_blob_mutation(self) -> None:
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.kind is ToolKind.BLOB_MUTATION

    def test_declaration_blob_kwarg_shape(self) -> None:
        """apply_pipeline_recipe may persist a recipe-scaffolded blob (needs quota+provenance) but
        advances CompositionState (NOT blob-store-only)."""
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.needs_blob_quota is True
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.needs_blob_provenance is True
        assert _APPLY_PIPELINE_RECIPE_DECLARATION.blob_store_only is False


class TestStep2RegistryAggregation:
    """All five blob-mutation declarations are aggregated into _REGISTERED_TOOLS at import time."""

    def test_all_blob_mutation_tools_are_declared(self) -> None:
        from elspeth.web.composer.tools._dispatch import _REGISTERED_TOOLS

        declared = {d.name for d in _REGISTERED_TOOLS if d.kind is ToolKind.BLOB_MUTATION}
        expected = {
            "create_blob",
            "update_blob",
            "delete_blob",
            "set_source_from_blob",
            "apply_pipeline_recipe",
        }
        assert declared == expected

    def test_registered_tools_count_is_five(self) -> None:
        from elspeth.web.composer.tools._dispatch import _REGISTERED_TOOLS

        # Step 2 should register exactly the five blob-mutation tools; no other tier yet.
        assert len(_REGISTERED_TOOLS) == 5


class TestToolDeclarationInvariants:
    """The constructor must crash early on inconsistent declarations."""

    @staticmethod
    def _make(**overrides: object) -> ToolDeclaration:
        defaults: dict[str, object] = {
            "name": "test_tool",
            "handler": _execute_create_blob,  # any callable with the right signature
            "kind": ToolKind.DISCOVERY,
            "description": "A test tool.",
            "json_schema": {"type": "object", "properties": {}, "required": []},
        }
        defaults.update(overrides)
        return ToolDeclaration(**defaults)  # type: ignore[arg-type]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            self._make(name="")

    def test_non_toolkind_kind_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a ToolKind member"):
            self._make(kind="discovery")  # type: ignore[arg-type]

    def test_cacheable_mutation_raises(self) -> None:
        with pytest.raises(ValueError, match="cacheable=True is forbidden"):
            self._make(kind=ToolKind.MUTATION, cacheable=True)

    def test_cacheable_session_aware_raises(self) -> None:
        with pytest.raises(ValueError, match="cacheable=True is forbidden"):
            self._make(kind=ToolKind.SESSION_AWARE, cacheable=True)

    def test_non_blob_mutation_with_blob_quota_raises(self) -> None:
        with pytest.raises(ValueError, match="needs_blob_quota=True is "):
            self._make(kind=ToolKind.MUTATION, needs_blob_quota=True)

    def test_non_blob_mutation_with_blob_provenance_raises(self) -> None:
        with pytest.raises(ValueError, match="needs_blob_provenance=True"):
            self._make(kind=ToolKind.DISCOVERY, needs_blob_provenance=True)

    def test_non_blob_mutation_with_blob_store_only_raises(self) -> None:
        with pytest.raises(ValueError, match="blob_store_only=True is "):
            self._make(kind=ToolKind.SECRET_MUTATION, blob_store_only=True)

    def test_json_schema_is_deeply_frozen(self) -> None:
        """Storing the json_schema deep-freezes — mutation of source dict cannot bleed in."""
        source: dict[str, object] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        decl = self._make(json_schema=source)
        # Mutate source post-construction; the declaration must be unaffected.
        source["required"] = ["y"]
        # deep_freeze converts the inner list to a tuple.
        assert decl.json_schema["required"] == ("x",)  # type: ignore[index]

    def test_emission_deep_thaws_back_to_json_shape(self) -> None:
        """derive_tool_definitions_by_name unfreezes so external consumers see JSON shapes."""
        from elspeth.web.composer.tools.declarations import derive_tool_definitions_by_name

        decl = self._make(
            json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
        )
        emitted = derive_tool_definitions_by_name([decl])[decl.name]
        # Parameters value is a plain dict; inner list is a plain list.
        assert isinstance(emitted["parameters"], dict)
        assert emitted["parameters"]["required"] == ["x"]  # not a tuple


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
        result = derive_tool_definitions_by_name([_CREATE_BLOB_DECLARATION])
        assert result["create_blob"] == _EXPECTED_CREATE_BLOB_DEFINITION

    def test_blob_quota_names_includes_create_blob(self) -> None:
        assert derive_blob_quota_names([_CREATE_BLOB_DECLARATION]) == {"create_blob"}

    def test_blob_provenance_names_includes_create_blob(self) -> None:
        assert derive_blob_provenance_names([_CREATE_BLOB_DECLARATION]) == {"create_blob"}

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
