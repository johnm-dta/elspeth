"""Tests for the shared plugin manager singleton."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from elspeth.plugins.infrastructure.manager import PluginManager, get_shared_plugin_manager


class TestGetSharedPluginManager:
    """Tests for get_shared_plugin_manager()."""

    def test_returns_plugin_manager_instance(self) -> None:
        pm = get_shared_plugin_manager()
        assert isinstance(pm, PluginManager)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        pm1 = get_shared_plugin_manager()
        pm2 = get_shared_plugin_manager()
        assert pm1 is pm2

    def test_has_builtin_sources_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_sources()) > 0

    def test_has_builtin_transforms_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_transforms()) > 0

    def test_has_builtin_sinks_registered(self) -> None:
        pm = get_shared_plugin_manager()
        assert len(pm.get_sinks()) > 0

    def test_failed_registration_does_not_poison_singleton(self) -> None:
        """If register_builtin_plugins() throws, the global must NOT be set.

        Regression test for elspeth-851bb384fb: previously, a failed registration
        left a half-initialized manager in _shared_instance, so all subsequent
        calls silently reused the broken instance.
        """
        import elspeth.plugins.infrastructure.manager as mgr_mod

        original = mgr_mod._shared_instance
        try:
            # Reset to force re-initialization
            mgr_mod._shared_instance = None

            with (
                patch.object(PluginManager, "register_builtin_plugins", side_effect=RuntimeError("boom")),
                pytest.raises(RuntimeError, match="boom"),
            ):
                get_shared_plugin_manager()

            # The global must still be None — not a broken manager
            assert mgr_mod._shared_instance is None

            # Next call without the mock should succeed
            pm = get_shared_plugin_manager()
            assert isinstance(pm, PluginManager)
            assert len(pm.get_sources()) > 0
        finally:
            mgr_mod._shared_instance = original
