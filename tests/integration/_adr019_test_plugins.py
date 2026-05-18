"""Config-dict-compatible fixture plugins for ADR-019 integration tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from elspeth.contracts import Determinism, SourceRow
from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl
from elspeth.plugins.infrastructure.manager import PluginManager
from tests.fixtures.plugins import (
    CollectSink,
    ConditionalErrorTransform,
    DivertingSink,
    ListSource,
)


class ADR019ListSource(ListSource):
    name = "list_source"
    determinism = Determinism.IO_READ

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        rows = options.pop("rows")
        super().__init__(
            list(rows),
            name=str(options.pop("name", self.name)),
            on_success=str(options.pop("on_success", "default")),
        )


class ADR019QuarantineSource(ListSource):
    name = "quarantine_source"
    determinism = Determinism.IO_READ

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        rows = options.pop("rows")
        super().__init__(
            [],
            name=str(options.pop("name", self.name)),
            on_success=str(options.pop("on_success", "default")),
        )
        self._quarantine_rows = list(rows)
        self._quarantine_destination = str(options.pop("quarantine_destination", "quarantine"))
        self._on_validation_failure = str(options.pop("on_validation_failure", "quarantine"))

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        del ctx
        for idx, row in enumerate(self._quarantine_rows):
            yield SourceRow.quarantined(
                row=dict(row),
                error=f"adr019 forced quarantine {idx}",
                destination=self._quarantine_destination,
            )


class ADR019CollectSink(CollectSink):
    name = "collect_sink"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        super().__init__(
            str(options.pop("name", self.name)),
            node_id=options.pop("node_id", None),
        )


class ADR019JsonCollectSink(ADR019CollectSink):
    """Collect sink registered as ``json`` for production failsink validation."""

    name = "json"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        CollectSink.__init__(
            self,
            self.name,
            node_id=options.pop("node_id", None),
        )


class ADR019ConditionalErrorTransform(ConditionalErrorTransform):
    name = "conditional_error"
    determinism = Determinism.DETERMINISTIC

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        super().__init__(
            name=options.pop("name", None),
            input_connection=options.pop("input", None),
            on_success=options.pop("on_success", None),
            on_error=options.pop("on_error", None),
        )


class ADR019DivertingSink(DivertingSink):
    name = "diverting_sink"


def make_adr019_plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager.register(create_dynamic_hookimpl([ADR019ListSource, ADR019QuarantineSource], "elspeth_get_source"))
    manager.register(create_dynamic_hookimpl([ADR019ConditionalErrorTransform], "elspeth_get_transforms"))
    manager.register(create_dynamic_hookimpl([ADR019CollectSink, ADR019JsonCollectSink, ADR019DivertingSink], "elspeth_get_sinks"))
    return manager


def install_adr019_test_plugin_manager(monkeypatch: pytest.MonkeyPatch) -> PluginManager:
    manager = make_adr019_plugin_manager()
    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: manager,
    )
    return manager
