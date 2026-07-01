"""Tests for inline ``plugin_schemas`` augmentation on failed option-shape mutations.

When a mutation tool (``set_pipeline``, ``upsert_node``, ``set_source``,
``set_output``, ``patch_*_options``, ``set_source_from_blob``,
``apply_pipeline_recipe``) returns ``success=False`` with at least one
``Invalid options for <kind> '<plugin>'`` validation error, the response
must embed the full ``get_plugin_schema`` payload for every named plugin
under a top-level ``plugin_schemas`` field. Eliminates the second LLM
round-trip the model would otherwise burn calling ``get_plugin_schema``
separately after each rejection (composer session 47cfbb5e on staging:
13 tool calls / 18 LLM rounds for a 4-plugin pipeline because the model
never preloaded any schema).
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSecretRequirement,
    PluginSummary,
)
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
)
from elspeth.web.composer.tools import execute_tool


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_catalog_with_schemas(
    source_schemas: dict[str, PluginSchemaInfo] | None = None,
    transform_schemas: dict[str, PluginSchemaInfo] | None = None,
    sink_schemas: dict[str, PluginSchemaInfo] | None = None,
) -> MagicMock:
    """Build a MagicMock catalog whose ``get_schema`` dispatches per (kind, name)."""
    source_schemas = source_schemas or {}
    transform_schemas = transform_schemas or {}
    sink_schemas = sink_schemas or {}

    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name=name,
            description=f"{name} source",
            plugin_type="source",
            config_fields=[
                ConfigFieldSummary(name="path", type="string", required=True, description="File path", default=None),
            ],
        )
        for name in (source_schemas or {"csv": None})
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name=name,
            description=f"{name} transform",
            plugin_type="transform",
            config_fields=[],
        )
        for name in (transform_schemas or {"passthrough": None})
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name=name,
            description=f"{name} sink",
            plugin_type="sink",
            config_fields=[],
        )
        for name in (sink_schemas or {"csv": None})
    ]

    def _dispatch_schema(plugin_type: Literal["source", "transform", "sink"], name: str) -> PluginSchemaInfo:
        bucket: dict[str, PluginSchemaInfo]
        if plugin_type == "source":
            bucket = source_schemas
        elif plugin_type == "transform":
            bucket = transform_schemas
        else:
            bucket = sink_schemas
        if name not in bucket:
            return PluginSchemaInfo(
                name=name,
                plugin_type=plugin_type,
                description=f"{name} {plugin_type}",
                json_schema={"title": f"{name.title()}Config", "properties": {}},
                knob_schema={"fields": []},
            )
        return bucket[name]

    catalog.get_schema.side_effect = _dispatch_schema
    return catalog


def _csv_schema() -> PluginSchemaInfo:
    return PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={
            "title": "CsvSourceConfig",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        knob_schema={"fields": [{"name": "path", "type": "string", "required": True}]},
    )


def _json_sink_schema() -> PluginSchemaInfo:
    return PluginSchemaInfo(
        name="json",
        plugin_type="sink",
        description="JSON sink",
        json_schema={
            "title": "JsonSinkConfig",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        knob_schema={"fields": [{"name": "path", "type": "string", "required": True}]},
    )


def _passthrough_transform_schema() -> PluginSchemaInfo:
    return PluginSchemaInfo(
        name="passthrough",
        plugin_type="transform",
        description="Passthrough transform",
        json_schema={"title": "PassthroughConfig", "properties": {}},
        knob_schema={"fields": []},
    )


def _azure_prompt_shield_schema() -> PluginSchemaInfo:
    return PluginSchemaInfo(
        name="azure_prompt_shield",
        plugin_type="transform",
        description="Prompt injection shield",
        json_schema={
            "title": "AzurePromptShieldConfig",
            "properties": {
                "endpoint": {"type": "string"},
                "api_key": {"type": "string"},
                "fields": {"type": "string"},
                "schema": {"type": "object"},
            },
            "required": ["endpoint", "api_key", "fields", "schema"],
        },
        knob_schema={"fields": []},
        secret_requirements=(PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)),),
    )


class TestFailureSchemaAugmentationSetPipeline:
    def test_failed_set_pipeline_includes_plugin_schemas_for_named_plugins(self) -> None:
        """set_pipeline rejection naming a plugin → that plugin's schema inline.

        set_pipeline is documented atomic: it bails at the first
        per-component validation failure and does not gather every
        option-shape rejection across source / nodes / sink. So this
        end-to-end test asserts schemas inline for whichever plugin
        the handler surfaced first. Multi-plugin error handling is
        covered separately by
        ``test_plugin_schemas_deduplicated_when_multiple_errors_name_same_plugin``
        and ``test_synthetic_multi_plugin_errors_produce_both_schemas``
        which feed a hand-crafted ValidationSummary through
        ``build_plugin_schemas_for_failure`` to prove the iteration
        contract.
        """
        catalog = _make_catalog_with_schemas(
            source_schemas={"csv": _csv_schema()},
            sink_schemas={"json": _json_sink_schema()},
        )
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                # Missing required ``path`` field — triggers Invalid options for source 'csv'.
                "options": {"schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [],
            "edges": [],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "json",
                    # Missing required ``path`` field — triggers Invalid options for sink 'json'.
                    "options": {"schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
        }

        result = execute_tool("set_pipeline", args, _empty_state(), catalog)
        payload = result.to_dict()

        assert result.success is False
        # set_pipeline is atomic — it stops at the first per-component
        # rejection it builds. The contract for plugin_schemas is "every
        # plugin named in a surfaced 'Invalid options for ...' error",
        # so we assert the exact iteration order that actually appears.
        assert "plugin_schemas" in payload
        # Build named_kinds as an ordered list (dict iteration is insertion
        # order in CPython 3.7+) so the assertion below is order-sensitive.
        named_kinds = [key.split("/", 1)[0] for key in payload["plugin_schemas"]]
        # set_pipeline reports the source schema first — this ordering is
        # the surface contract relied on by composer prompt construction
        # (prompts.py consumes plugin_schemas in iteration order). If this
        # flips to sink-first, or if multiple kinds appear (meaning atomic
        # bail-at-first-failure was relaxed), the LLM context will reorder;
        # reassess prompts.py before changing this expectation.
        assert named_kinds == ["source"]
        for key, schema in payload["plugin_schemas"].items():
            kind, plugin = key.split("/", 1)
            assert schema["name"] == plugin
            assert schema["plugin_type"] == kind
            assert "json_schema" in schema

    def test_successful_set_pipeline_omits_plugin_schemas(self, tmp_path: Any) -> None:
        """A successful mutation must NOT carry the optional plugin_schemas field."""
        catalog = _make_catalog_with_schemas(
            source_schemas={"csv": _csv_schema()},
            transform_schemas={"passthrough": _passthrough_transform_schema()},
            sink_schemas={
                "csv": PluginSchemaInfo(
                    name="csv",
                    plugin_type="sink",
                    description="CSV sink",
                    json_schema={"title": "CsvSinkConfig", "properties": {}},
                    knob_schema={"fields": []},
                )
            },
        )
        args = {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": str(tmp_path / "blobs" / "in.csv"), "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            "nodes": [
                {
                    "id": "t1",
                    "node_type": "transform",
                    "plugin": "passthrough",
                    "input": "rows",
                    "on_success": "main",
                    "on_error": "discard",
                    "options": {"schema": {"mode": "observed"}},
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "from_node": "source",
                    "to_node": "t1",
                    "edge_type": "on_success",
                    "label": None,
                }
            ],
            "outputs": [
                {
                    "sink_name": "main",
                    "plugin": "csv",
                    "options": {
                        "path": str(tmp_path / "outputs" / "out.csv"),
                        "schema": {"mode": "observed"},
                        "mode": "write",
                        "collision_policy": "auto_increment",
                    },
                    "on_write_failure": "discard",
                }
            ],
        }

        result = execute_tool("set_pipeline", args, _empty_state(), catalog, data_dir=str(tmp_path))
        payload = result.to_dict()

        assert result.success is True, payload
        assert "plugin_schemas" not in payload


class TestFailureSchemaAugmentationMultiPluginErrors:
    """Cover the multi-plugin iteration contract directly.

    ``set_pipeline`` is atomic and bails at the first per-component
    rejection, so it cannot in practice surface "Invalid options for
    source 'csv'" AND "Invalid options for sink 'json'" in a single
    response. The augmentation hook is nonetheless built to iterate
    every error entry — when a future mutation tool *does* aggregate
    rejections (or a synthetic ToolResult carries multiple), every
    distinct ``(kind, plugin)`` pair must materialise in
    ``plugin_schemas``. These tests feed a hand-crafted ValidationSummary
    through ``build_plugin_schemas_for_failure`` to lock that contract
    independently of the per-tool execution paths.
    """

    def test_synthetic_multi_plugin_errors_produce_both_schemas(self) -> None:
        """Two errors naming different plugins → both schemas inline."""
        from elspeth.web.composer.state import ValidationEntry, ValidationSummary
        from elspeth.web.composer.tools._common import (
            ToolResult,
            build_plugin_schemas_for_failure,
        )

        validation = ValidationSummary(
            is_valid=False,
            errors=(
                ValidationEntry(
                    component="rejected_mutation",
                    message=(
                        "Invalid options for source 'csv': missing required field 'path'. "
                        "Also: Invalid options for sink 'json': missing required field 'path'."
                    ),
                    severity="high",
                ),
            ),
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=validation,
            affected_nodes=(),
        )
        catalog = _make_catalog_with_schemas(
            source_schemas={"csv": _csv_schema()},
            sink_schemas={"json": _json_sink_schema()},
        )

        schemas = build_plugin_schemas_for_failure(result, catalog)

        assert schemas is not None
        assert list(schemas.keys()) == ["sink/json", "source/csv"]
        assert schemas["source/csv"]["plugin_type"] == "source"
        assert schemas["sink/json"]["plugin_type"] == "sink"


class TestFailureSchemaAugmentationDeduplication:
    def test_plugin_schemas_deduplicated_when_multiple_errors_name_same_plugin(self) -> None:
        """Two distinct errors naming the same (kind, plugin) → schema emitted once."""
        from elspeth.web.composer.state import ValidationEntry, ValidationSummary
        from elspeth.web.composer.tools._common import (
            ToolResult,
            build_plugin_schemas_for_failure,
        )

        # Build a synthetic ToolResult containing two errors that both
        # name source 'csv' under the option-shape pattern.
        validation = ValidationSummary(
            is_valid=False,
            errors=(
                ValidationEntry(
                    component="rejected_mutation",
                    message="Invalid options for source 'csv': missing required field 'path'",
                    severity="high",
                ),
                ValidationEntry(
                    component="source",
                    message="Invalid options for source 'csv': trailing detail",
                    severity="high",
                ),
            ),
        )
        result = ToolResult(
            success=False,
            updated_state=_empty_state(),
            validation=validation,
            affected_nodes=(),
        )
        catalog = _make_catalog_with_schemas(source_schemas={"csv": _csv_schema()})

        schemas = build_plugin_schemas_for_failure(result, catalog)
        assert schemas is not None
        assert list(schemas.keys()) == ["source/csv"]


class TestFailureSchemaAugmentationPerToolCoverage:
    """Confirm the augmentation hook fires for every option-shape tool."""

    def test_set_source_failure_carries_schema(self) -> None:
        catalog = _make_catalog_with_schemas(source_schemas={"csv": _csv_schema()})
        # Missing required ``path`` triggers Invalid options for source 'csv'.
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            _empty_state(),
            catalog,
        )
        payload = result.to_dict()
        assert result.success is False
        assert "plugin_schemas" in payload
        assert "source/csv" in payload["plugin_schemas"]

    def test_set_output_failure_carries_schema(self) -> None:
        catalog = _make_catalog_with_schemas(sink_schemas={"json": _json_sink_schema()})
        # Missing required ``path`` triggers Invalid options for sink 'json'.
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "json",
                "options": {"schema": {"mode": "observed"}, "mode": "write", "collision_policy": "auto_increment"},
                "on_write_failure": "discard",
            },
            _empty_state(),
            catalog,
        )
        payload = result.to_dict()
        assert result.success is False
        assert "plugin_schemas" in payload
        assert "sink/json" in payload["plugin_schemas"]

    def test_upsert_node_failure_carries_schema(self) -> None:
        """upsert_node for a transform with missing required ``schema`` triggers Invalid options."""
        # ``passthrough`` (via TransformDataConfig) requires a schema
        # field on options. Omitting it produces the option-shape
        # rejection the augmentation hook is built for.
        catalog = _make_catalog_with_schemas(
            transform_schemas={"passthrough": _passthrough_transform_schema()},
        )
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "rows",
                "on_success": "main",
                "on_error": "discard",
                "options": {},  # missing required ``schema``
            },
            _empty_state(),
            catalog,
        )
        payload = result.to_dict()
        assert result.success is False, payload
        errors_text = " ".join(e["message"] for e in payload["validation"]["errors"])
        assert "Invalid options for transform 'passthrough'" in errors_text, errors_text
        assert "plugin_schemas" in payload, payload
        assert "transform/passthrough" in payload["plugin_schemas"]

    def test_unavailable_secret_required_transform_schema_is_not_inlined(self) -> None:
        """Augmentation must not bypass get_plugin_schema's secret-availability gate."""
        catalog = _make_catalog_with_schemas(
            transform_schemas={"azure_prompt_shield": _azure_prompt_shield_schema()},
        )
        secret_service = MagicMock()
        secret_service.has_ref.return_value = False

        result = execute_tool(
            "upsert_node",
            {
                "id": "shield",
                "node_type": "transform",
                "plugin": "azure_prompt_shield",
                "input": "rows",
                "on_success": "llm",
                "on_error": "discard",
                "options": {},
            },
            _empty_state(),
            catalog,
            secret_service=secret_service,
            user_id="test-user",
        )
        payload = result.to_dict()

        assert result.success is False, payload
        errors_text = " ".join(e["message"] for e in payload["validation"]["errors"])
        assert "Invalid options for transform 'azure_prompt_shield'" in errors_text
        assert "plugin_schemas" not in payload

    def test_patch_source_options_failure_carries_schema(self) -> None:
        """patch_source_options surfacing Invalid options for source 'csv' must carry the schema inline."""
        catalog = _make_catalog_with_schemas(
            source_schemas={"csv": _csv_schema()},
        )

        # Stage a valid source via set_source first.
        set_result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "rows",
                "options": {"path": "/data/blobs/in.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "discard",
            },
            _empty_state(),
            catalog,
            data_dir="/data",
        )
        assert set_result.success is True, set_result.to_dict()

        # patch_source_options that nulls out the required ``path`` field
        # — _apply_merge_patch drops None-valued keys, so the resulting
        # options dict loses ``path`` and _prevalidate_source rejects
        # via the same Invalid options path the augmentation hook
        # covers. The wrapper model names the key ``patch``.
        result = execute_tool(
            "patch_source_options",
            {
                "patch": {"path": None},
            },
            set_result.updated_state,
            catalog,
        )
        payload = result.to_dict()
        assert result.success is False, payload
        errors_text = " ".join(e["message"] for e in payload["validation"]["errors"])
        assert "Invalid options for source 'csv'" in errors_text, errors_text
        assert "plugin_schemas" in payload, payload
        assert "source/csv" in payload["plugin_schemas"]


class TestFailureSchemaAugmentationNonAugmentedTools:
    def test_get_plugin_schema_does_not_carry_plugin_schemas_field(self) -> None:
        """Discovery tools — even on failure — must NOT trigger augmentation."""
        catalog = _make_catalog_with_schemas()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: nope")
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "nope"},
            _empty_state(),
            catalog,
        )
        payload = result.to_dict()
        assert result.success is False
        assert "plugin_schemas" not in payload
