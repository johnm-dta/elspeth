"""Low-layer source/sink role helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, SinkContext, SourceContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.plugin_roles import (
    require_declared_input_fields_plugin,
    require_declared_output_fields_plugin,
    sink_declared_required_fields,
    source_declared_guaranteed_fields,
)
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource


def _contract(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=str,
                required=True,
                source="inferred",
                nullable=False,
            )
            for name in fields
        ),
        locked=True,
    )


class _DeclaredSourceBase(BaseSource):
    name = "declared-source-base"
    determinism = Determinism.IO_READ
    output_schema = PluginSchema
    declared_guaranteed_fields = frozenset({"customer_id"})

    def __init__(self) -> None:
        self.config = {}
        self.node_id = None

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        yield SourceRow.valid({"customer_id": "v"}, contract=_contract(("customer_id",)), source_row_index=0)

    def close(self) -> None:
        pass


class _InheritedDeclaredSource(_DeclaredSourceBase):
    determinism = Determinism.IO_READ
    pass


class _BadDeclaredSourceTuple(_DeclaredSourceBase):
    determinism = Determinism.IO_READ
    declared_guaranteed_fields = ("customer_id",)


class _BadDeclaredSourceItem(_DeclaredSourceBase):
    determinism = Determinism.IO_READ
    declared_guaranteed_fields = frozenset({1})


class _DeclaredSinkBase(BaseSink):
    name = "declared-sink-base"
    determinism = Determinism.IO_WRITE
    input_schema = PluginSchema
    declared_required_fields = frozenset({"customer_id"})

    def __init__(self) -> None:
        self.config = {}
        self.node_id = None

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _InheritedDeclaredSink(_DeclaredSinkBase):
    determinism = Determinism.IO_WRITE
    pass


class _BadDeclaredSinkTuple(_DeclaredSinkBase):
    determinism = Determinism.IO_WRITE
    declared_required_fields = ("customer_id",)


class _BadDeclaredSinkItem(_DeclaredSinkBase):
    determinism = Determinism.IO_WRITE
    declared_required_fields = frozenset({1})


class _DeclaredOutputPlugin:
    name: Any = "declared-output"
    node_id: Any = "node-1"
    declared_output_fields: Any = frozenset({"customer_id"})


class _BadDeclaredOutputTuple(_DeclaredOutputPlugin):
    declared_output_fields = ("customer_id",)


class _BadDeclaredOutputItem(_DeclaredOutputPlugin):
    declared_output_fields = frozenset({1})


class _BadDeclaredOutputName(_DeclaredOutputPlugin):
    name = ""


class _BadDeclaredOutputNodeID(_DeclaredOutputPlugin):
    node_id = object()


class _DeclaredInputPlugin:
    name: Any = "declared-input"
    node_id: Any = "node-1"
    declared_input_fields: Any = frozenset({"customer_id"})
    is_batch_aware: Any = False


class _BadDeclaredInputTuple(_DeclaredInputPlugin):
    declared_input_fields = ("customer_id",)


class _BadDeclaredInputItem(_DeclaredInputPlugin):
    declared_input_fields = frozenset({1})


class _BadDeclaredInputBatchFlag(_DeclaredInputPlugin):
    is_batch_aware = "false"


class _BadDeclaredInputName(_DeclaredInputPlugin):
    name = ""


def test_source_declared_guaranteed_fields_uses_inherited_source_declaration() -> None:
    assert source_declared_guaranteed_fields(_InheritedDeclaredSource()) == frozenset({"customer_id"})


def test_source_declared_guaranteed_fields_rejects_non_source_even_when_attr_present() -> None:
    class _NotASource:
        name = "not-a-source"
        node_id = None
        declared_guaranteed_fields = frozenset({"customer_id"})

    assert source_declared_guaranteed_fields(_NotASource()) is None


def test_sink_declared_required_fields_uses_inherited_sink_declaration() -> None:
    assert sink_declared_required_fields(_InheritedDeclaredSink()) == frozenset({"customer_id"})


def test_sink_declared_required_fields_rejects_non_sink_even_when_attr_present() -> None:
    class _NotASink:
        name = "not-a-sink"
        node_id = None
        declared_required_fields = frozenset({"customer_id"})

    assert sink_declared_required_fields(_NotASink()) is None


@pytest.mark.parametrize(
    ("plugin", "expected"),
    [
        (_InheritedDeclaredSource(), frozenset({"customer_id"})),
        (_InheritedDeclaredSink(), None),
    ],
)
def test_source_helper_does_not_cross_match_sink_role(
    plugin: object,
    expected: frozenset[str] | None,
) -> None:
    assert source_declared_guaranteed_fields(plugin) == expected


@pytest.mark.parametrize(
    ("plugin", "expected"),
    [
        (_InheritedDeclaredSink(), frozenset({"customer_id"})),
        (_InheritedDeclaredSource(), None),
    ],
)
def test_sink_helper_does_not_cross_match_source_role(
    plugin: object,
    expected: frozenset[str] | None,
) -> None:
    assert sink_declared_required_fields(plugin) == expected


@pytest.mark.parametrize(
    ("helper", "plugin", "match"),
    [
        (
            source_declared_guaranteed_fields,
            _BadDeclaredSourceTuple(),
            r"_BadDeclaredSourceTuple\.declared_guaranteed_fields must be frozenset, got 'tuple'\.",
        ),
        (
            source_declared_guaranteed_fields,
            _BadDeclaredSourceItem(),
            r"_BadDeclaredSourceItem\.declared_guaranteed_fields must contain only str items\.",
        ),
        (
            sink_declared_required_fields,
            _BadDeclaredSinkTuple(),
            r"_BadDeclaredSinkTuple\.declared_required_fields must be frozenset, got 'tuple'\.",
        ),
        (
            sink_declared_required_fields,
            _BadDeclaredSinkItem(),
            r"_BadDeclaredSinkItem\.declared_required_fields must contain only str items\.",
        ),
    ],
)
def test_role_helpers_reject_invalid_declared_field_surfaces(
    helper: Callable[[object], frozenset[str] | None],
    plugin: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        helper(plugin)


@pytest.mark.parametrize(
    ("plugin", "match"),
    [
        (
            _BadDeclaredOutputTuple(),
            "declared_output_fields must be frozenset",
        ),
        (
            _BadDeclaredOutputItem(),
            "declared_output_fields must contain only str items",
        ),
        (
            _BadDeclaredOutputName(),
            "name must be a non-empty str",
        ),
        (
            _BadDeclaredOutputNodeID(),
            "node_id must be str",
        ),
    ],
)
def test_require_declared_output_fields_plugin_rejects_invalid_contract_surface(
    plugin: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        require_declared_output_fields_plugin(plugin)


@pytest.mark.parametrize(
    ("plugin", "match"),
    [
        (
            _BadDeclaredInputTuple(),
            "declared_input_fields must be frozenset",
        ),
        (
            _BadDeclaredInputItem(),
            "declared_input_fields must contain only str items",
        ),
        (
            _BadDeclaredInputBatchFlag(),
            "is_batch_aware must be bool",
        ),
        (
            _BadDeclaredInputName(),
            "name must be a non-empty str",
        ),
    ],
)
def test_require_declared_input_fields_plugin_rejects_invalid_contract_surface(
    plugin: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        require_declared_input_fields_plugin(plugin)
