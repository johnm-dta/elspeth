"""Tests for FastAPI dependency injection providers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from elspeth.web.config import WebSettings
from elspeth.web.dependencies import (
    get_auth_provider,
    get_session_service,
    get_settings,
)


def _request_with_state(**state_attrs: object) -> SimpleNamespace:
    """Build the app.state shape used by FastAPI Request dependencies."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state_attrs)))


class _EmptyPluginManager:
    def get_sources(self) -> list[object]:
        return []

    def get_transforms(self) -> list[object]:
        return []

    def get_sinks(self) -> list[object]:
        return []


class TestGetSettings:
    def test_returns_settings_from_app_state(self) -> None:
        settings = WebSettings(
            composer_max_composition_turns=1,
            composer_max_discovery_turns=1,
            composer_timeout_seconds=1.0,
            composer_rate_limit_per_minute=1,
            shareable_link_signing_key=b"\x00" * 32,
        )
        request = _request_with_state(settings=settings)
        assert get_settings(request) is settings


class TestGetSessionService:
    def test_returns_session_service_from_app_state(self) -> None:
        service = object()
        request = _request_with_state(session_service=service)
        assert get_session_service(request) is service


class TestGetAuthProvider:
    def test_returns_auth_provider_from_app_state(self) -> None:
        provider = object()
        request = _request_with_state(auth_provider=provider)
        assert get_auth_provider(request) is provider


class TestCreateCatalogService:
    def test_returns_catalog_service_instance(self) -> None:
        from elspeth.web.catalog.service import CatalogServiceImpl
        from elspeth.web.dependencies import create_catalog_service

        manager = _EmptyPluginManager()

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=manager):
            service = create_catalog_service()

        assert isinstance(service, CatalogServiceImpl)
