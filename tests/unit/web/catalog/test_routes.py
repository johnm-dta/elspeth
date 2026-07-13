"""Tests for catalog API routes via TestClient."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from elspeth.plugins.infrastructure.manager import PluginManager, get_shared_plugin_manager
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.routes import catalog_router
from elspeth.web.catalog.service import CatalogServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def catalog(plugin_manager: PluginManager) -> CatalogServiceImpl:
    return CatalogServiceImpl(plugin_manager)


@pytest.fixture
def client(catalog: CatalogServiceImpl) -> TestClient:
    """TestClient with catalog router mounted."""
    app = FastAPI()
    app.state.catalog_service = catalog
    settings = WebSettings(
        composer_max_composition_turns=4,
        composer_max_discovery_turns=4,
        composer_timeout_seconds=60,
        composer_rate_limit_per_minute=20,
        shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
        llm_profiles={
            "task-role": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)

    class _Inventory:
        def has_server_ref(self, name: str) -> bool:
            return False

        def has_user_ref(self, principal: str, name: str) -> bool:
            return False

        def has_ref(self, principal: str, name: str) -> bool:
            return False

    app.state.web_plugin_policy = policy
    app.state.operator_profile_registry = profiles
    app.state.plugin_snapshot_factory = lambda user: build_plugin_snapshot(
        policy=policy,
        catalog=catalog,
        profiles=profiles,
        principal_scope=f"local:{user.user_id}",
        secret_inventory=_Inventory(),
        generation_key=b"catalog-route-test-key",
    )
    app.dependency_overrides[get_current_user] = lambda: UserIdentity(user_id="alice", username="alice")
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

    def test_required_field_mapper_present(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms")
        names = [e["name"] for e in resp.json()]
        assert "field_mapper" in names
        assert "passthrough" not in names

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

    def test_field_mapper_transform_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/transforms/field_mapper/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "field_mapper"
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

    def test_disabled_null_source_schema_is_hidden(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/null/schema")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "plugin_not_enabled"

    def test_unknown_type_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/widgets/csv/schema")
        assert resp.status_code == 404
        assert "Unknown plugin type" in resp.json()["detail"]

    def test_unknown_name_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/nonexistent_xyz/schema")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "plugin_not_enabled"

    def test_unknown_name_includes_available_list(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sources/nonexistent_xyz/schema")
        assert "Available:" not in resp.json()["detail"]

    def test_disabled_direct_schema_cannot_enumerate_plugin(self, client: TestClient) -> None:
        response = client.get("/api/catalog/transforms/azure_prompt_shield/schema")
        assert response.status_code == 404
        assert "azure_prompt_shield" not in response.json()["detail"]


def test_policy_and_catalog_responses_are_private_and_fingerprinted(client: TestClient) -> None:
    for path in ("/api/catalog/policy", "/api/catalog/sources", "/api/catalog/transforms/llm/schema"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["vary"] == "Authorization, Cookie"
        assert response.headers["x-elspeth-plugin-snapshot"]

    policy = client.get("/api/catalog/policy").json()
    assert "snapshot_fingerprint" in policy
    assert "transform:llm" in policy["available_plugin_ids"]
