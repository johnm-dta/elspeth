"""Enforcement test tying the shared observed-text contract rule to the plugin.

The shared contract helpers in elspeth.contracts.schema infer that an observed
text source with column='X' guarantees field 'X'. TextSource.__init__ must make
that guarantee visible in both its normalized schema config and the raw config
dict consumed by DAG construction. This test verifies both sides agree: the
plugin produces the key and declares it as guaranteed for observed schemas only.
If either side changes, this test fails.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.contexts import SourceContext
from elspeth.plugins.sources.text_source import TextSource


class TestTextSourceHeuristicEnforcement:
    def test_text_source_produces_configured_column_key(self) -> None:
        """Text source output key must match the shared contract rule."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
            source = TextSource(config)
            ctx = MagicMock(spec=SourceContext)
            ctx.record_validation_error = MagicMock()

            rows = list(source.load(ctx))

            assert len(rows) >= 1
            first_row = rows[0].row
            assert "text" in first_row, (
                "TextSource with column='text' must produce rows with key 'text'. "
                f"Got keys: {list(first_row.keys())}. "
                "The shared contract helper in elspeth.contracts.schema depends on this."
            )
            assert first_row["text"] == "hello"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_text_source_auto_declares_guaranteed_fields(self) -> None:
        """Observed text sources must auto-declare {column} in both config views."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
            source = TextSource(config)

            assert source._schema_config.guaranteed_fields == ("text",), (
                "TextSource must set the exact observed-text guarantee on its "
                "normalized SchemaConfig using mechanical dataclass state, not "
                "only via a derived effective-guarantees view."
            )
            raw_schema = source.config["schema"]
            assert raw_schema["guaranteed_fields"] == ["text"], (
                "TextSource must also write the observed-text guarantee back to source.config['schema'] so DAG construction sees it."
            )
            assert config["schema"] == {"mode": "observed"}, (
                "TextSource must not mutate the caller-supplied raw config when "
                "it backfills the observed-text guarantee for runtime DAG use."
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_text_source_preserves_explicit_guaranteed_fields(self) -> None:
        """Explicit guaranteed_fields must not be overridden."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": {"mode": "observed", "guaranteed_fields": ["custom_field"]},
                "on_validation_failure": "quarantine",
            }
            source = TextSource(config)

            guaranteed = source._schema_config.get_effective_guaranteed_fields()
            assert "custom_field" in guaranteed, "Explicit guaranteed_fields must be preserved"
            assert "text" not in guaranteed
            assert source.config == config, (
                "Observed schemas with explicit guaranteed_fields must keep the "
                "raw source config unchanged so source node IDs remain stable."
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        ("schema", "field_name"),
        [
            ({"mode": "fixed", "fields": ["text: str"]}, "text"),
            ({"mode": "flexible", "fields": ["text: str"]}, "text"),
        ],
    )
    def test_text_source_observed_only_auto_declare_does_not_touch_non_observed_schema(
        self,
        schema: dict[str, Any],
        field_name: str,
    ) -> None:
        """Fixed/flexible schemas must keep raw config identity and implicit guarantees."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            tmp_path = f.name

        try:
            config: dict[str, Any] = {
                "path": tmp_path,
                "column": "text",
                "schema": schema,
                "on_validation_failure": "quarantine",
            }
            source = TextSource(config)

            assert source._schema_config.declares_guaranteed_fields is False, (
                "declares_guaranteed_fields tracks explicit guaranteed_fields "
                "only. Fixed typed fields remain implicit guarantees and must "
                "not be rewritten into an explicit observed-text declaration."
            )
            assert field_name in source._schema_config.get_effective_guaranteed_fields()
            assert source.config == config, (
                "Non-observed text source configs must remain unchanged. "
                "Rewriting them changes DAG node IDs even when "
                "schema semantics did not change."
            )
            assert "guaranteed_fields" not in source.config["schema"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)
