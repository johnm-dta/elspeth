"""Tests for ChromaSink configuration validation."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.chroma_sink import ChromaSinkConfig


class TestFieldMappingConfig:
    """Tests for field_mapping validation rules."""

    def test_valid_config(self) -> None:
        config = ChromaSinkConfig.from_dict(
            {
                "collection": "science-facts",
                "mode": "persistent",
                "persist_directory": "./chroma_data",
                "distance_function": "cosine",
                "field_mapping": {
                    "document_field": "text_content",
                    "id_field": "doc_id",
                    "metadata_fields": ["topic", "subtopic"],
                },
                "on_duplicate": "overwrite",
                "schema": {
                    "mode": "fixed",
                    "fields": [
                        "doc_id: str",
                        "text_content: str",
                        "topic: str",
                        "subtopic: str",
                    ],
                },
            }
        )
        assert config.collection == "science-facts"
        assert config.field_mapping.document_field == "text_content"
        assert config.field_mapping.id_field == "doc_id"
        assert config.field_mapping.metadata_fields == ("topic", "subtopic")

    def test_field_mapping_required(self) -> None:
        with pytest.raises(Exception, match="field_mapping"):
            ChromaSinkConfig.from_dict(
                {
                    "collection": "test",
                    "mode": "persistent",
                    "persist_directory": "./data",
                    "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
                }
            )

    def test_on_duplicate_default_is_overwrite(self) -> None:
        config = ChromaSinkConfig.from_dict(
            {
                "collection": "test",
                "mode": "persistent",
                "persist_directory": "./data",
                "field_mapping": {"document_field": "text", "id_field": "id", "metadata_fields": []},
                "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
            }
        )
        assert config.on_duplicate == "overwrite"

    def test_on_duplicate_rejects_invalid_value(self) -> None:
        with pytest.raises(PluginConfigError, match="on_duplicate"):
            ChromaSinkConfig.from_dict(
                {
                    "collection": "test",
                    "mode": "persistent",
                    "persist_directory": "./data",
                    "field_mapping": {
                        "document_field": "text",
                        "id_field": "id",
                        "metadata_fields": [],
                    },
                    "on_duplicate": "invalid",
                    "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
                }
            )

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(PluginConfigError, match="extra"):
            ChromaSinkConfig.from_dict(
                {
                    "collection": "test",
                    "mode": "persistent",
                    "persist_directory": "./data",
                    "field_mapping": {
                        "document_field": "text",
                        "id_field": "id",
                        "metadata_fields": [],
                    },
                    "unknown_extra": "value",
                    "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
                }
            )


class TestFieldMappingSchemaValidation:
    """Tests that field_mapping references are validated against schema at config time."""

    @staticmethod
    def _make_config(
        fields: list[str],
        *,
        mode: str = "fixed",
        document_field: str = "text",
        id_field: str = "id",
        metadata_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "collection": "test",
            "mode": "persistent",
            "persist_directory": "./data",
            "field_mapping": {
                "document_field": document_field,
                "id_field": id_field,
                "metadata_fields": metadata_fields or [],
            },
            "schema": {"mode": mode, "fields": fields} if mode != "observed" else {"mode": "observed"},
        }

    def test_valid_fixed_schema(self) -> None:
        config = ChromaSinkConfig.from_dict(
            self._make_config(
                ["id: str", "text: str", "score: float"],
                metadata_fields=["score"],
            )
        )
        assert config.field_mapping.metadata_fields == ("score",)

    def test_document_field_not_in_schema(self) -> None:
        with pytest.raises(PluginConfigError, match=r"document_field.*'text'.*not in the schema"):
            ChromaSinkConfig.from_dict(self._make_config(["id: str", "body: str"], document_field="text"))

    def test_id_field_not_in_schema(self) -> None:
        with pytest.raises(PluginConfigError, match=r"id_field.*'doc_id'.*not in the schema"):
            ChromaSinkConfig.from_dict(self._make_config(["id: str", "text: str"], id_field="doc_id"))

    def test_metadata_field_not_in_schema(self) -> None:
        with pytest.raises(PluginConfigError, match=r"metadata_fields.*'missing'.*not in the schema"):
            ChromaSinkConfig.from_dict(self._make_config(["id: str", "text: str"], metadata_fields=["missing"]))

    def test_document_field_wrong_type(self) -> None:
        with pytest.raises(PluginConfigError, match=r"document_field.*'text'.*type 'int'.*requires str"):
            ChromaSinkConfig.from_dict(self._make_config(["id: str", "text: int"], document_field="text"))

    def test_id_field_wrong_type(self) -> None:
        with pytest.raises(PluginConfigError, match=r"id_field.*'id'.*type 'float'.*requires str"):
            ChromaSinkConfig.from_dict(self._make_config(["id: float", "text: str"], id_field="id"))

    def test_any_type_accepted_for_document_field(self) -> None:
        """'any' type is compatible with str — runtime validates actual values."""
        config = ChromaSinkConfig.from_dict(self._make_config(["id: any", "text: any"]))
        assert config.field_mapping.document_field == "text"

    def test_all_chroma_metadata_types_accepted(self) -> None:
        """ChromaDB accepts str, int, float, bool as metadata values."""
        config = ChromaSinkConfig.from_dict(
            self._make_config(
                ["id: str", "text: str", "name: str", "count: int", "score: float", "active: bool"],
                metadata_fields=["name", "count", "score", "active"],
            )
        )
        assert len(config.field_mapping.metadata_fields) == 4

    def test_observed_schema_skips_validation(self) -> None:
        """Observed mode has no field definitions — validation defers to runtime."""
        config = ChromaSinkConfig.from_dict(self._make_config([], mode="observed", metadata_fields=["anything"]))
        assert config.schema_config.is_observed

    def test_flexible_schema_validates(self) -> None:
        """Flexible mode has declared fields — validation applies."""
        with pytest.raises(PluginConfigError, match=r"document_field.*'missing'.*not in the schema"):
            ChromaSinkConfig.from_dict(
                self._make_config(
                    ["id: str", "text: str"],
                    mode="flexible",
                    document_field="missing",
                )
            )

    def test_empty_metadata_fields_accepted(self) -> None:
        """No metadata fields is a valid configuration."""
        config = ChromaSinkConfig.from_dict(self._make_config(["id: str", "text: str"], metadata_fields=[]))
        assert config.field_mapping.metadata_fields == ()


class TestConnectionValidation:
    """Tests that ChromaConnectionConfig validation flows through."""

    def test_persistent_mode_requires_persist_directory(self) -> None:
        with pytest.raises(Exception, match="persist_directory"):
            ChromaSinkConfig.from_dict(
                {
                    "collection": "test",
                    "mode": "persistent",
                    "field_mapping": {"document_field": "t", "id_field": "i", "metadata_fields": []},
                    "schema": {"mode": "fixed", "fields": ["i: str", "t: str"]},
                }
            )

    def test_client_mode_requires_host(self) -> None:
        with pytest.raises(Exception, match="host"):
            ChromaSinkConfig.from_dict(
                {
                    "collection": "test",
                    "mode": "client",
                    "field_mapping": {"document_field": "t", "id_field": "i", "metadata_fields": []},
                    "schema": {"mode": "fixed", "fields": ["i: str", "t: str"]},
                }
            )
