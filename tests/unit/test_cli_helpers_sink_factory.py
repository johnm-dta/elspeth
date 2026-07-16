"""Tests for make_sink_factory() — fresh sink instances from config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest


@dataclass(frozen=True, slots=True)
class _SinkConfigFake:
    plugin: str
    options: dict[str, Any]
    on_write_failure: str


@dataclass(frozen=True, slots=True)
class _ConfigFake:
    sinks: dict[str, _SinkConfigFake]


class _SinkFake:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self._on_write_failure = "fail"


class _SinkClassRecorder:
    def __init__(self) -> None:
        self.instances: list[_SinkFake] = []

    def __call__(self, options: dict[str, Any]) -> _SinkFake:
        instance = _SinkFake(options)
        self.instances.append(instance)
        return instance


class _PluginManagerFake:
    def __init__(self, sink_cls: type[_SinkFake] | _SinkClassRecorder) -> None:
        self._sink_cls = sink_cls

    def get_sink_by_name(self, _name: str) -> type[_SinkFake] | _SinkClassRecorder:
        return self._sink_cls


class TestMakeSinkFactory:
    """Tests for runtime_factory.make_sink_factory()."""

    def _make_config_with_sink(self, sink_name: str = "csv_out", plugin_name: str = "csv", on_write_failure: str = "fail") -> _ConfigFake:
        sink_config = _SinkConfigFake(
            plugin=plugin_name,
            options={"path": "/tmp/out.csv"},
            on_write_failure=on_write_failure,
        )
        return _ConfigFake(sinks={sink_name: sink_config})

    def test_raises_on_unknown_sink_name(self) -> None:
        from elspeth.plugins.infrastructure.runtime_factory import make_sink_factory

        config = self._make_config_with_sink("csv_out")
        factory = make_sink_factory(config)

        with pytest.raises(ValueError, match="not found in sink configuration"):
            factory("nonexistent")

    def test_copies_on_write_failure(self) -> None:
        from elspeth.plugins.infrastructure.runtime_factory import make_sink_factory

        config = self._make_config_with_sink(on_write_failure="quarantine")

        sink_cls = _SinkClassRecorder()
        manager = _PluginManagerFake(sink_cls)

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=manager):
            factory = make_sink_factory(config)
            binding = factory("csv_out")

        assert binding.sink._on_write_failure == "quarantine"

    def test_returns_fresh_instances(self) -> None:
        from elspeth.plugins.infrastructure.runtime_factory import make_sink_factory

        config = self._make_config_with_sink()

        manager = _PluginManagerFake(_SinkFake)

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=manager):
            factory = make_sink_factory(config)
            sink1 = factory("csv_out").sink
            sink2 = factory("csv_out").sink

        assert sink1 is not sink2
