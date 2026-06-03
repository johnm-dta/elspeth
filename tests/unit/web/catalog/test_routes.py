"""Tests for catalog API routes via TestClient."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.routes import catalog_router
from elspeth.web.catalog.service import CatalogServiceImpl
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def catalog(plugin_manager: PluginManager) -> CatalogServiceImpl:
    return CatalogServiceImpl(plugin_manager)


@pytest.fixture
def client(catalog: CatalogServiceImpl) -> TestClient:
    """TestClient with catalog router mounted."""
    app = FastAPI()
    app.state.catalog_service = catalog
    app.include_router(catalog_router, prefix="/api/catalog")
    return TestClient(app)


class TestListSources:
    """GET /api/catalog/sources"""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources")
        assert resp.status_code == 200

    def test_returns_json_array(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources")
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_each_entry_has_required_fields(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources")
        for entry in resp.json():
            assert "name" in entry
            assert "description" in entry
            assert "plugin_type" in entry
            assert "config_fields" in entry
            assert entry["plugin_type"] == "source"

    def test_csv_source_present(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources")
        names = [e["name"] for e in resp.json()]
        assert "csv" in names

    def test_text_source_present(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources")
        names = [e["name"] for e in resp.json()]
        assert "text" in names

    def test_csv_source_summary_includes_reference_content(self, client: TestClient) -> None:
        """Wire-shape pin: catalog API returns canonical CSV reference content."""
        resp = client.get("/api/catalog/sources")
        assert resp.status_code == 200
        sources = resp.json()
        csv = next(s for s in sources if s["name"] == "csv")
        assert csv["usage_when_to_use"] is not None
        assert "tabular" in csv["capability_tags"]
        # io_read is the kind-default determinism for sources; the catalog
        # suppresses default-derived flags so the strip only shows author
        # decisions. csv inherits the default, so no determinism flag.
        assert "io_read" not in csv["audit_characteristics"]
        assert "coerce" in csv["audit_characteristics"]  # author-declared
        assert "quarantine" in csv["audit_characteristics"]  # author-declared


class TestListTransforms:
    """GET /api/catalog/transforms"""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms")
        assert resp.status_code == 200

    def test_returns_json_array(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms")
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_passthrough_present(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms")
        names = [e["name"] for e in resp.json()]
        assert "passthrough" in names

    def test_all_entries_have_transform_type(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms")
        entries = resp.json()
        assert len(entries) > 0, "catalog returned no entries; type assertion would be vacuously true"
        for entry in entries:
            assert entry["plugin_type"] == "transform"


class TestListSinks:
    """GET /api/catalog/sinks"""

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sinks")
        assert resp.status_code == 200

    def test_csv_sink_present(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sinks")
        names = [e["name"] for e in resp.json()]
        assert "csv" in names


class TestGetSchema:
    """GET /api/catalog/{type}/{name}/schema"""

    def test_csv_source_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/csv/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "csv"
        assert data["plugin_type"] == "source"
        assert "json_schema" in data
        assert "properties" in data["json_schema"]

    def test_passthrough_transform_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms/passthrough/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "passthrough"
        assert data["plugin_type"] == "transform"

    def test_csv_sink_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sinks/csv/schema")
        assert resp.status_code == 200

    def test_text_source_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/text/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "text"
        assert data["plugin_type"] == "source"

    def test_null_source_returns_empty_schema(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/null/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["json_schema"] == {}

    def test_unknown_type_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/widgets/csv/schema")
        assert resp.status_code == 404
        assert "Unknown plugin type" in resp.json()["detail"]

    def test_unknown_name_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/nonexistent_xyz/schema")
        assert resp.status_code == 404
        assert "Unknown source plugin" in resp.json()["detail"]

    def test_unknown_name_includes_available_list(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/nonexistent_xyz/schema")
        assert "Available:" in resp.json()["detail"]
