"""Runtime source guaranteed-field declaration tests (ADR-016)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from elspeth.contracts import SourceRow
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.sources.text_source import TextSource


class _StubSource(BaseSource):
    name = "stub"

    def __init__(self, schema_config: SchemaConfig) -> None:
        super().__init__({})
        self._on_validation_failure = "discard"
        self._initialize_declared_guaranteed_fields(schema_config)
        from elspeth.contracts.data import PluginSchema

        self.output_schema = PluginSchema

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        return iter([])

    def close(self) -> None:
        pass


def test_base_source_helper_uses_effective_guarantees_for_observed_schema() -> None:
    source = _StubSource(
        SchemaConfig.from_dict(
            {
                "mode": "observed",
                "guaranteed_fields": ["customer_id", "account_id"],
            }
        )
    )
    assert source.declared_guaranteed_fields == frozenset({"customer_id", "account_id"})


def test_base_source_helper_uses_required_declared_fields_for_fixed_schema() -> None:
    source = _StubSource(
        SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": [
                    "id: str",
                    "name: str?",
                ],
            }
        )
    )
    assert source.declared_guaranteed_fields == frozenset({"id"})


def test_base_source_helper_returns_empty_when_schema_abstains() -> None:
    source = _StubSource(SchemaConfig.from_dict({"mode": "observed"}))
    assert source.declared_guaranteed_fields == frozenset()


def test_text_source_runtime_declaration_reflects_observed_mode_column_heuristic(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("hello\n", encoding="utf-8")

    source = TextSource(
        {
            "path": str(path),
            "column": "body",
            "schema": {"mode": "observed"},
            "on_validation_failure": "discard",
        }
    )

    assert source.declared_guaranteed_fields == frozenset({"body"})


def test_null_source_keeps_empty_runtime_guarantee_surface() -> None:
    source = NullSource({"on_success": "default"})
    assert source.declared_guaranteed_fields == frozenset()
