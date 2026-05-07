"""Tests for FastAPI dependency injection providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from elspeth.web.config import WebSettings
from elspeth.web.dependencies import (
    get_auth_provider,
    get_session_service,
    get_settings,
)


def _mock_request(**state_attrs: object) -> MagicMock:
    """Build a mock Request with app.state attributes."""
    request = MagicMock()
    for key, value in state_attrs.items():
        setattr(request.app.state, key, value)
    return request


class TestGetSettings:
    def test_returns_settings_from_app_state(self) -> None:
        settings = MagicMock(spec=WebSettings)
        request = _mock_request(settings=settings)
        assert get_settings(request) is settings


class TestGetSessionService:
    def test_returns_session_service_from_app_state(self) -> None:
        service = MagicMock()
        request = _mock_request(session_service=service)
        assert get_session_service(request) is service


class TestGetAuthProvider:
    def test_returns_auth_provider_from_app_state(self) -> None:
        provider = MagicMock()
        request = _mock_request(auth_provider=provider)
        assert get_auth_provider(request) is provider


class TestCreateCatalogService:
    def test_returns_catalog_service_instance(self) -> None:
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.web.catalog.service import CatalogServiceImpl
        from elspeth.web.dependencies import create_catalog_service

        mock_manager = MagicMock(spec=PluginManager)
        mock_manager.get_sources.return_value = []
        mock_manager.get_transforms.return_value = []
        mock_manager.get_sinks.return_value = []

        with patch("elspeth.plugins.infrastructure.manager.get_shared_plugin_manager", return_value=mock_manager):
            service = create_catalog_service()

        assert isinstance(service, CatalogServiceImpl)
