from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.knob_schema import KnobSchema, KnobSchemaLoweringError
from elspeth.web.catalog.service import CatalogServiceImpl


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    pm = PluginManager()
    pm.register_builtin_plugins()
    return pm


def test_catalog_lowering_runs_at_init_not_first_request(plugin_manager: PluginManager) -> None:
    svc = CatalogServiceImpl(plugin_manager)

    info = svc.get_schema("source", "csv")

    assert "knob_schema" in info.model_dump()
    assert info.knob_schema["fields"]


def test_catalog_get_schema_reads_cache(plugin_manager: PluginManager, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def counting_lower(*args: Any, **kwargs: Any) -> KnobSchema:
        nonlocal calls
        calls += 1
        return {"fields": []}

    monkeypatch.setattr("elspeth.web.catalog.service.lower_model_to_knob_schema", counting_lower)
    svc = CatalogServiceImpl(plugin_manager)
    calls_after_init = calls

    svc.get_schema("source", "csv")
    svc.get_schema("source", "csv")

    assert calls_after_init > 0
    assert calls == calls_after_init


def test_catalog_init_raises_when_lowering_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenOptions(BaseModel):
        value: str

    class _BrokenSource:
        name = "broken_source"
        config_model = _BrokenOptions

        @classmethod
        def get_config_model(cls, config: dict[str, Any] | None = None) -> type[BaseModel]:
            del config
            return cls.config_model

        @classmethod
        def get_config_schema(cls) -> dict[str, Any]:
            return cls.config_model.model_json_schema()

    class _FakePluginManager:
        def get_sources(self) -> list[type[_BrokenSource]]:
            return [_BrokenSource]

        def get_transforms(self) -> list[Any]:
            return []

        def get_sinks(self) -> list[Any]:
            return []

    def failing_lower(*args: Any, **kwargs: Any) -> KnobSchema:
        raise KnobSchemaLoweringError(
            plugin_kind="source",
            plugin_name="broken_source",
            field_path="value",
            constraint="intentional lowering failure",
            remediation="fix the plugin metadata",
        )

    monkeypatch.setattr("elspeth.web.catalog.service.lower_model_to_knob_schema", failing_lower)

    with pytest.raises(KnobSchemaLoweringError):
        CatalogServiceImpl(_FakePluginManager())  # type: ignore[arg-type]
