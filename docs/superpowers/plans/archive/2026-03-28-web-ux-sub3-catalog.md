# Web UX Sub-Plan 3: Catalog — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only plugin catalog browsing to the web application. Wrap `PluginManager` behind a `CatalogService` protocol, expose four REST endpoints, and serialize Pydantic config schemas to JSON.

**Architecture:** `CatalogServiceImpl` receives an already-initialized `PluginManager` via constructor injection (the shared singleton from `get_shared_plugin_manager()`). It does NOT call `register_builtin_plugins()` itself -- the singleton factory handles initialization. Schema resolution delegates to `PluginConfigValidator` to avoid duplicating the name-to-config-class mapping. All endpoints are read-only. Cache populated once at startup.

**Tech Stack:** FastAPI, Pydantic v2 (response models + `model_json_schema()`), pluggy (via PluginManager)

**Spec:** `docs/superpowers/specs/2026-03-28-web-ux-sub3-catalog-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/catalog/__init__.py` | Module init |
| Create | `src/elspeth/web/catalog/protocol.py` | `CatalogService` protocol (4 methods) |
| Create | `src/elspeth/web/catalog/schemas.py` | `ConfigFieldSummary`, `PluginSummary`, `PluginSchemaInfo` response models |
| Create | `src/elspeth/web/catalog/service.py` | `CatalogServiceImpl` wrapping `PluginManager` |
| Create | `src/elspeth/web/catalog/routes.py` | FastAPI router with 4 GET endpoints |
| Modify | `src/elspeth/web/app.py` | Register catalog router in app factory |
| Modify | `src/elspeth/web/dependencies.py` | Add `CatalogService` dependency provider |
| Create | `tests/unit/web/catalog/__init__.py` | Test module init |
| Create | `tests/unit/web/catalog/test_service.py` | CatalogServiceImpl tests with real `PluginManager` |
| Create | `tests/unit/web/catalog/test_routes.py` | Catalog API endpoint tests via `TestClient` |

---

### Task 3.1: CatalogService Protocol, Schemas, and Implementation

**Files:**
- Create: `src/elspeth/web/catalog/__init__.py`
- Create: `src/elspeth/web/catalog/protocol.py`
- Create: `src/elspeth/web/catalog/schemas.py`
- Create: `src/elspeth/web/catalog/service.py`
- Create: `tests/unit/web/catalog/__init__.py`
- Create: `tests/unit/web/catalog/test_service.py`

- [ ] **Step 1: Create module init and response schemas**

```python
# src/elspeth/web/catalog/__init__.py
```

```python
# src/elspeth/web/catalog/schemas.py
"""Catalog API response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigFieldSummary(BaseModel):
    """Summary of a single field in a plugin's config model."""

    name: str
    type: str
    required: bool
    description: str | None = None
    default: Any | None = None


class PluginSummary(BaseModel):
    """Lightweight plugin info for catalog browsing."""

    name: str
    description: str
    plugin_type: str
    config_fields: list[ConfigFieldSummary]


class PluginSchemaInfo(BaseModel):
    """Full plugin schema detail for the composer."""

    name: str
    plugin_type: str
    description: str
    json_schema: dict[str, Any]
```

- [ ] **Step 2: Create the CatalogService protocol**

```python
# src/elspeth/web/catalog/protocol.py
"""CatalogService protocol — internal service boundary for plugin catalog."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary


@runtime_checkable
class CatalogService(Protocol):
    """Read-only plugin catalog.

    Four methods, all synchronous (plugin discovery is CPU-bound).
    When the Catalog module is later extracted to a microservice,
    this protocol stays and the implementation becomes an HTTP client.
    """

    def list_sources(self) -> list[PluginSummary]: ...

    def list_transforms(self) -> list[PluginSummary]: ...

    def list_sinks(self) -> list[PluginSummary]: ...

    def get_schema(self, plugin_type: str, name: str) -> PluginSchemaInfo: ...
```

- [ ] **Step 3: Write CatalogServiceImpl tests**

Tests use real `PluginManager` with `register_builtin_plugins()` -- no mocked plugin lists.

```python
# tests/unit/web/catalog/__init__.py
```

```python
# tests/unit/web/catalog/test_service.py
"""Tests for CatalogServiceImpl with real PluginManager."""

from __future__ import annotations

import pytest

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    """Shared PluginManager with builtins registered."""
    pm = PluginManager()
    pm.register_builtin_plugins()
    return pm


@pytest.fixture(scope="module")
def catalog(plugin_manager: PluginManager) -> CatalogServiceImpl:
    return CatalogServiceImpl(plugin_manager)


class TestCatalogServiceProtocol:
    """CatalogServiceImpl satisfies the CatalogService protocol."""

    def test_implements_protocol(self, catalog: CatalogServiceImpl) -> None:
        assert isinstance(catalog, CatalogService)


class TestListSources:
    """list_sources() returns all registered source plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        assert len(sources) > 0

    def test_csv_source_present(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        names = [s.name for s in sources]
        assert "csv" in names

    def test_all_entries_are_plugin_summaries(
        self, catalog: CatalogServiceImpl
    ) -> None:
        sources = catalog.list_sources()
        for s in sources:
            assert isinstance(s, PluginSummary)
            assert s.plugin_type == "source"
            assert s.name
            assert s.description

    def test_config_fields_populated_for_csv(
        self, catalog: CatalogServiceImpl
    ) -> None:
        sources = catalog.list_sources()
        csv_source = next(s for s in sources if s.name == "csv")
        field_names = [f.name for f in csv_source.config_fields]
        # CSVSourceConfig has a 'path' field at minimum
        assert "path" in field_names

    def test_config_field_has_type_and_required(
        self, catalog: CatalogServiceImpl
    ) -> None:
        sources = catalog.list_sources()
        csv_source = next(s for s in sources if s.name == "csv")
        for field in csv_source.config_fields:
            assert field.type  # non-empty type string
            assert isinstance(field.required, bool)

    def test_matches_plugin_manager_count(
        self,
        catalog: CatalogServiceImpl,
        plugin_manager: PluginManager,
    ) -> None:
        sources = catalog.list_sources()
        assert len(sources) == len(plugin_manager.get_sources())


class TestListTransforms:
    """list_transforms() returns all registered transform plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        assert len(transforms) > 0

    def test_passthrough_present(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        names = [t.name for t in transforms]
        assert "passthrough" in names

    def test_all_entries_have_transform_type(
        self, catalog: CatalogServiceImpl
    ) -> None:
        transforms = catalog.list_transforms()
        for t in transforms:
            assert t.plugin_type == "transform"

    def test_matches_plugin_manager_count(
        self,
        catalog: CatalogServiceImpl,
        plugin_manager: PluginManager,
    ) -> None:
        transforms = catalog.list_transforms()
        assert len(transforms) == len(plugin_manager.get_transforms())


class TestListSinks:
    """list_sinks() returns all registered sink plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        sinks = catalog.list_sinks()
        assert len(sinks) > 0

    def test_csv_sink_present(self, catalog: CatalogServiceImpl) -> None:
        sinks = catalog.list_sinks()
        names = [s.name for s in sinks]
        assert "csv" in names

    def test_all_entries_have_sink_type(
        self, catalog: CatalogServiceImpl
    ) -> None:
        sinks = catalog.list_sinks()
        for s in sinks:
            assert s.plugin_type == "sink"

    def test_matches_plugin_manager_count(
        self,
        catalog: CatalogServiceImpl,
        plugin_manager: PluginManager,
    ) -> None:
        sinks = catalog.list_sinks()
        assert len(sinks) == len(plugin_manager.get_sinks())


class TestGetSchema:
    """get_schema() returns full JSON schema for a plugin's config."""

    def test_csv_source_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("source", "csv")
        assert isinstance(info, PluginSchemaInfo)
        assert info.name == "csv"
        assert info.plugin_type == "source"
        assert info.description  # non-empty
        assert isinstance(info.json_schema, dict)
        # Pydantic JSON schema has 'properties' and 'type'
        assert "properties" in info.json_schema
        assert info.json_schema["type"] == "object"

    def test_passthrough_transform_schema(
        self, catalog: CatalogServiceImpl
    ) -> None:
        info = catalog.get_schema("transform", "passthrough")
        assert info.name == "passthrough"
        assert info.plugin_type == "transform"
        assert isinstance(info.json_schema, dict)

    def test_csv_sink_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("sink", "csv")
        assert info.name == "csv"
        assert info.plugin_type == "sink"
        assert isinstance(info.json_schema, dict)

    def test_null_source_returns_empty_schema(
        self, catalog: CatalogServiceImpl
    ) -> None:
        info = catalog.get_schema("source", "null")
        assert info.name == "null"
        assert info.json_schema == {}

    def test_llm_transform_returns_base_schema(
        self, catalog: CatalogServiceImpl
    ) -> None:
        info = catalog.get_schema("transform", "llm")
        assert info.name == "llm"
        assert isinstance(info.json_schema, dict)
        # Base LLMConfig has a 'provider' field
        if info.json_schema:
            assert "properties" in info.json_schema

    def test_unknown_type_raises_value_error(
        self, catalog: CatalogServiceImpl
    ) -> None:
        with pytest.raises(ValueError, match="Unknown plugin type"):
            catalog.get_schema("widgets", "csv")

    def test_unknown_name_raises_value_error(
        self, catalog: CatalogServiceImpl
    ) -> None:
        with pytest.raises(ValueError, match="Unknown source plugin"):
            catalog.get_schema("source", "nonexistent_plugin_xyz")

    def test_unknown_name_includes_available(
        self, catalog: CatalogServiceImpl
    ) -> None:
        with pytest.raises(ValueError, match="Available:"):
            catalog.get_schema("source", "nonexistent_plugin_xyz")
```

- [ ] **Step 4: Implement CatalogServiceImpl**

```python
# src/elspeth/web/catalog/service.py
"""CatalogServiceImpl — wraps PluginManager for catalog browsing."""

from __future__ import annotations

from typing import Any

from elspeth.plugins.infrastructure.discovery import get_plugin_description
from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.plugins.infrastructure.validation import PluginConfigValidator
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSummary,
)

# Valid plugin type path segments and their PluginManager lookup methods
_VALID_TYPES = frozenset({"source", "transform", "sink"})


class CatalogServiceImpl:
    """Read-only catalog backed by PluginManager.

    Receives an already-initialized PluginManager via constructor injection.
    Does NOT call register_builtin_plugins() -- the shared singleton factory
    (get_shared_plugin_manager) handles initialization before injection.

    Caches plugin class lists once at construction. The plugin set is
    fixed for the lifetime of the process.
    """

    def __init__(self, plugin_manager: PluginManager) -> None:
        self._pm = plugin_manager
        self._validator = PluginConfigValidator()

        # Cache plugin classes once
        self._source_classes = plugin_manager.get_sources()
        self._transform_classes = plugin_manager.get_transforms()
        self._sink_classes = plugin_manager.get_sinks()

    def list_sources(self) -> list[PluginSummary]:
        return [
            self._to_summary(cls, "source") for cls in self._source_classes
        ]

    def list_transforms(self) -> list[PluginSummary]:
        return [
            self._to_summary(cls, "transform")
            for cls in self._transform_classes
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [
            self._to_summary(cls, "sink") for cls in self._sink_classes
        ]

    def get_schema(
        self, plugin_type: str, name: str
    ) -> PluginSchemaInfo:
        if plugin_type not in _VALID_TYPES:
            raise ValueError(
                f"Unknown plugin type: {plugin_type}. "
                f"Must be one of: {sorted(_VALID_TYPES)}"
            )

        # Look up plugin class to verify it exists
        lookup = {
            "source": self._pm.get_source_by_name,
            "transform": self._pm.get_transform_by_name,
            "sink": self._pm.get_sink_by_name,
        }
        try:
            plugin_cls = lookup[plugin_type](name)
        except ValueError:
            # Re-raise with catalog-specific message format
            available = self._available_names(plugin_type)
            raise ValueError(
                f"Unknown {plugin_type} plugin: {name}. "
                f"Available: {available}"
            ) from None

        # Get config model via PluginConfigValidator
        json_schema = self._get_json_schema(plugin_type, name)

        # Full docstring for schema view (not just first line)
        description = (plugin_cls.__doc__ or "").strip()
        if not description:
            description = get_plugin_description(plugin_cls)

        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=description,
            json_schema=json_schema,
        )

    # -- Private helpers --

    def _to_summary(self, plugin_cls: type, plugin_type: str) -> PluginSummary:
        """Convert a plugin class to a PluginSummary."""
        name: str = plugin_cls.name  # type: ignore[attr-defined]
        description = get_plugin_description(plugin_cls)
        config_fields = self._extract_config_fields(plugin_type, name)
        return PluginSummary(
            name=name,
            description=description,
            plugin_type=plugin_type,
            config_fields=config_fields,
        )

    def _extract_config_fields(
        self, plugin_type: str, name: str
    ) -> list[ConfigFieldSummary]:
        """Extract config field summaries from a plugin's Pydantic config model."""
        # Map singular plugin_type to validator method
        config_model = self._resolve_config_model(plugin_type, name)
        if config_model is None:
            return []

        schema = config_model.model_json_schema()
        properties: dict[str, Any] = schema.get("properties", {})
        required_fields: set[str] = set(schema.get("required", []))

        fields: list[ConfigFieldSummary] = []
        for field_name, field_schema in properties.items():
            json_type = field_schema.get("type", "object")
            # anyOf produces no top-level type — pick first branch type
            if "anyOf" in field_schema and not field_schema.get("type"):
                for branch in field_schema["anyOf"]:
                    if branch.get("type") != "null":
                        json_type = branch.get("type", "object")
                        break

            fields.append(
                ConfigFieldSummary(
                    name=field_name,
                    type=json_type,
                    required=field_name in required_fields,
                    description=field_schema.get("description"),
                    default=field_schema.get("default"),
                )
            )

        return fields

    def _resolve_config_model(
        self, plugin_type: str, name: str
    ) -> type | None:
        """Resolve plugin name to its Pydantic config model class.

        Delegates to PluginConfigValidator's private methods to avoid
        duplicating the name-to-config-class mapping.

        Returns None for plugins with no config model (e.g., null source).
        """
        try:
            if plugin_type in ("source", "sources"):
                return self._validator._get_source_config_model(name)
            elif plugin_type in ("transform", "transforms"):
                return self._validator._get_transform_config_model(name)
            elif plugin_type in ("sink", "sinks"):
                return self._validator._get_sink_config_model(name)
        except ValueError:
            # Plugin exists in PluginManager but has no config model mapping
            # in PluginConfigValidator — return None (empty schema)
            return None
        return None

    def _get_json_schema(
        self, plugin_type: str, name: str
    ) -> dict[str, Any]:
        """Get full JSON schema for a plugin's config model."""
        config_model = self._resolve_config_model(plugin_type, name)
        if config_model is None:
            return {}
        return config_model.model_json_schema()

    def _available_names(self, plugin_type: str) -> list[str]:
        """Get sorted list of available plugin names for a type."""
        classes = {
            "source": self._source_classes,
            "transform": self._transform_classes,
            "sink": self._sink_classes,
        }[plugin_type]
        return sorted(
            cls.name for cls in classes  # type: ignore[attr-defined]
        )
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/test_service.py -x -v
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(web/catalog): add CatalogService protocol and implementation"
```

---

### Task 3.2: Catalog API Routes

**Files:**
- Create: `src/elspeth/web/catalog/routes.py`
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/dependencies.py`
- Create: `tests/unit/web/catalog/test_routes.py`

- [ ] **Step 1: Write route tests**

```python
# tests/unit/web/catalog/test_routes.py
"""Tests for catalog API routes via TestClient."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.routes import catalog_router
from elspeth.web.catalog.service import CatalogServiceImpl


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    pm = PluginManager()
    pm.register_builtin_plugins()
    return pm


@pytest.fixture(scope="module")
def catalog(plugin_manager: PluginManager) -> CatalogServiceImpl:
    return CatalogServiceImpl(plugin_manager)


@pytest.fixture(scope="module")
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

    def test_each_entry_has_required_fields(
        self, client: TestClient
    ) -> None:
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

    def test_all_entries_have_transform_type(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/catalog/transforms")
        for entry in resp.json():
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

    def test_passthrough_transform_schema_200(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/catalog/transforms/passthrough/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "passthrough"
        assert data["plugin_type"] == "transform"

    def test_csv_sink_schema_200(self, client: TestClient) -> None:
        resp = client.get("/api/catalog/sinks/csv/schema")
        assert resp.status_code == 200

    def test_null_source_returns_empty_schema(
        self, client: TestClient
    ) -> None:
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

    def test_unknown_name_includes_available_list(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/catalog/sources/nonexistent_xyz/schema")
        assert "Available:" in resp.json()["detail"]
```

- [ ] **Step 2: Implement routes**

```python
# src/elspeth/web/catalog/routes.py
"""Catalog API routes — read-only plugin browsing."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary

catalog_router = APIRouter(tags=["catalog"])

# Map plural REST path segments to singular protocol values (M3 fix).
# The CatalogService protocol uses singular ("source", "transform", "sink").
# REST paths use plural ("sources", "transforms", "sinks").
# Translation is a REST presentation concern — does not leak into service layer.
_PLURAL_TO_SINGULAR = {"sources": "source", "transforms": "transform", "sinks": "sink"}


def _get_catalog(request: Request):  # noqa: ANN202
    """Extract CatalogService from app state."""
    return request.app.state.catalog_service


@catalog_router.get("/sources", response_model=list[PluginSummary])
def list_sources(request: Request) -> list[PluginSummary]:
    """List all registered source plugins."""
    return _get_catalog(request).list_sources()


@catalog_router.get("/transforms", response_model=list[PluginSummary])
def list_transforms(request: Request) -> list[PluginSummary]:
    """List all registered transform plugins."""
    return _get_catalog(request).list_transforms()


@catalog_router.get("/sinks", response_model=list[PluginSummary])
def list_sinks(request: Request) -> list[PluginSummary]:
    """List all registered sink plugins."""
    return _get_catalog(request).list_sinks()


@catalog_router.get(
    "/{plugin_type}/{name}/schema", response_model=PluginSchemaInfo
)
def get_schema(
    plugin_type: str, name: str, request: Request
) -> PluginSchemaInfo:
    """Get full JSON schema for a plugin's configuration."""
    singular = _PLURAL_TO_SINGULAR.get(plugin_type)
    if singular is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown plugin type: {plugin_type}. Must be one of: {sorted(_PLURAL_TO_SINGULAR)}",
        )
    try:
        return _get_catalog(request).get_schema(singular, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
```

- [ ] **Step 3: Wire into app factory and dependencies**

The `web/app.py` and `web/dependencies.py` files are created in Phase 1 (Foundation). The catalog additions are:

**In `src/elspeth/web/dependencies.py`** -- add the catalog service provider:

```python
# Add to dependencies.py

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl


def create_catalog_service() -> CatalogServiceImpl:
    """Create CatalogService backed by the shared PluginManager singleton.

    get_shared_plugin_manager() returns an already-initialized manager
    (with register_builtin_plugins() called). CatalogServiceImpl does
    not re-initialize -- it caches the existing plugin lists.
    """
    return CatalogServiceImpl(get_shared_plugin_manager())
```

**In `src/elspeth/web/app.py`** -- register the catalog router in `create_app()`:

```python
# Add to create_app() in app.py

from elspeth.web.catalog.routes import catalog_router
from elspeth.web.dependencies import create_catalog_service

# Inside create_app():
app.state.catalog_service = create_catalog_service()
app.include_router(catalog_router, prefix="/api/catalog")
```

- [ ] **Step 4: Run all catalog tests**

```bash
.venv/bin/python -m pytest tests/unit/web/catalog/ -x -v
```

- [ ] **Step 5: Run type checker and linter**

```bash
.venv/bin/python -m mypy src/elspeth/web/catalog/
.venv/bin/python -m ruff check src/elspeth/web/catalog/
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(web/catalog): add catalog API routes and wire into app factory"
```

---

## Round 4 Review Amendments

> **Status: Amendments below have been integrated into the plan body.**

- **R4-M3a: Plural-to-singular translation in route handler.** The `CatalogService` protocol uses singular `plugin_type` values (`"source"`, `"transform"`, `"sink"`), but the REST endpoints use plural path segments (`sources`, `transforms`, `sinks`). In Task 3.2 Step 2 (`catalog/routes.py`), the `get_schema()` route handler must translate the plural `plugin_type` path parameter to singular before calling `CatalogService.get_schema()`. Map: `{"sources": "source", "transforms": "transform", "sinks": "sink"}`. Return 404 if the path segment is not in the map.
- **R4-M3b: Seam contract C alignment.** The Catalog-to-Composer interface is defined in Seam C of `specs/2026-03-28-web-ux-seam-contracts.md`. Sub-4 (LLM Composer) calls `CatalogService` using singular forms matching the protocol. The plural-to-singular translation is purely a REST presentation concern and must not leak into the service layer.
