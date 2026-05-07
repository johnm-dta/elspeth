"""Tier 1 strictness regression tests for catalog response schemas.

The catalog API exposes plugin metadata — names, types, schemas — that
the frontend renders into configuration UI.  If a backend bug emitted
a wrong type (a ``bool`` where a ``str`` is expected, or an extra
undeclared field), a permissive ``BaseModel`` would silently forward
garbage to the UI.  These tests lock in the strict response contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSummary,
)


class TestCatalogStrictCoercionRejected:
    def test_config_field_summary_rejects_string_bool_required(self) -> None:
        with pytest.raises(ValidationError):
            ConfigFieldSummary(name="api_key", type="string", required="true")  # type: ignore[arg-type]

    def test_config_field_summary_rejects_int_for_name(self) -> None:
        with pytest.raises(ValidationError):
            ConfigFieldSummary(name=42, type="string", required=False)  # type: ignore[arg-type]

    def test_plugin_summary_rejects_invalid_plugin_type(self) -> None:
        """plugin_type is a Literal — "plugin" must be rejected."""
        with pytest.raises(ValidationError):
            PluginSummary(
                name="csv",
                description="d",
                plugin_type="plugin",  # type: ignore[arg-type]
                config_fields=[],
            )

    def test_plugin_schema_info_rejects_invalid_plugin_type(self) -> None:
        with pytest.raises(ValidationError):
            PluginSchemaInfo(
                name="csv",
                plugin_type="pipeline",  # type: ignore[arg-type]
                description="d",
                json_schema={},
            )


class TestCatalogExtraFieldsRejected:
    def test_config_field_summary_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ConfigFieldSummary(
                name="api_key",
                type="string",
                required=True,
                sensitive=True,  # type: ignore[call-arg]
            )

    def test_plugin_summary_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            PluginSummary(
                name="csv",
                description="d",
                plugin_type="source",
                config_fields=[],
                version="1.0",  # type: ignore[call-arg]
            )

    def test_plugin_schema_info_rejects_extra(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            PluginSchemaInfo(
                name="csv",
                plugin_type="source",
                description="d",
                json_schema={},
                deprecated=False,  # type: ignore[call-arg]
            )


class TestCatalogHappyPath:
    def test_config_field_summary_with_defaults(self) -> None:
        field = ConfigFieldSummary(name="api_key", type="string", required=True)
        assert field.description is None
        assert field.default is None

    def test_plugin_summary_accepts_any_kind(self) -> None:
        for kind in ("source", "transform", "sink"):
            resp = PluginSummary(
                name="p",
                description="d",
                plugin_type=kind,  # type: ignore[arg-type]
                config_fields=[],
            )
            assert resp.plugin_type == kind

    def test_plugin_schema_info_with_empty_schema(self) -> None:
        resp = PluginSchemaInfo(
            name="null",
            plugin_type="source",
            description="",
            json_schema={},
        )
        assert resp.json_schema == {}
