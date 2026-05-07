"""Tests for _make_sink_factory() — fresh sink instances from config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elspeth.plugins.infrastructure.manager import PluginManager


class TestMakeSinkFactory:
    """Tests for cli_helpers._make_sink_factory()."""

    def _make_config_with_sink(self, sink_name: str = "csv_out", plugin_name: str = "csv", on_write_failure: str = "fail") -> MagicMock:
        sink_config = MagicMock()
        sink_config.plugin = plugin_name
        sink_config.options = {"path": "/tmp/out.csv"}
        sink_config.on_write_failure = on_write_failure
        config = MagicMock()
        config.sinks = {sink_name: sink_config}
        return config

    def test_raises_on_unknown_sink_name(self) -> None:
        from elspeth.cli_helpers import _make_sink_factory

        config = self._make_config_with_sink("csv_out")
        factory = _make_sink_factory(config)

        with pytest.raises(ValueError, match="not found in sink configuration"):
            factory("nonexistent")

    def test_copies_on_write_failure(self) -> None:
        from elspeth.cli_helpers import _make_sink_factory

        config = self._make_config_with_sink(on_write_failure="quarantine")

        mock_sink_cls = MagicMock()
        mock_sink_instance = MagicMock()
        mock_sink_cls.return_value = mock_sink_instance

        mock_manager = MagicMock(spec=PluginManager)
        mock_manager.get_sink_by_name.return_value = mock_sink_cls

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=mock_manager):
            factory = _make_sink_factory(config)
            sink = factory("csv_out")

        assert sink._on_write_failure == "quarantine"

    def test_returns_fresh_instances(self) -> None:
        from elspeth.cli_helpers import _make_sink_factory

        config = self._make_config_with_sink()

        mock_manager = MagicMock(spec=PluginManager)
        mock_manager.get_sink_by_name.return_value = MagicMock

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=mock_manager):
            factory = _make_sink_factory(config)
            sink1 = factory("csv_out")
            sink2 = factory("csv_out")

        assert sink1 is not sink2
