# Web UX Sub-Spec 3: Catalog

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 3
**Depends On:** Sub-Spec 1 (Foundation)
**Blocks:** Sub-Spec 4 (Composer)

---

## Scope

This phase adds read-only plugin catalog browsing to the web application. It wraps the existing `PluginManager` (from `plugins/infrastructure/manager.py`) behind a `CatalogService` Protocol, exposes it via four REST endpoints, and serializes Pydantic config schemas to JSON for consumption by the frontend inspector and the LLM composer's discovery tools.

No mutation endpoints. No plugin instantiation. The catalog is a static, cacheable view of what plugins are available and what configuration each one accepts.

---

## CatalogService Protocol

The `CatalogService` protocol defines the internal service boundary. Four methods, all synchronous (plugin discovery is CPU-bound, not I/O-bound):

- `list_sources() -> list[PluginSummary]` -- All registered source plugins with name, description, and config field summary.
- `list_transforms() -> list[PluginSummary]` -- All registered transform plugins. Gates are excluded (they are config-driven system operations, not plugins).
- `list_sinks() -> list[PluginSummary]` -- All registered sink plugins.
- `get_schema(plugin_type: str, name: str) -> PluginSchemaInfo` -- Full Pydantic JSON schema for a specific plugin's configuration model. Raises `ValueError` if the plugin type or name is unknown.

`CatalogServiceImpl` is the sole implementation. It receives a `PluginManager` instance via constructor injection (the same instance wired through FastAPI's dependency injection in `dependencies.py`). On construction, it calls `PluginManager.register_builtin_plugins()` if the manager has not already been initialized, then caches the plugin class lists. The cache is populated once at startup and never invalidated -- the plugin set is fixed for the lifetime of the process.

The protocol lives in `web/catalog/protocol.py`. When the Catalog module is later extracted to a microservice, the protocol stays and the implementation becomes an HTTP client.

---

## Plugin Discovery

Plugin discovery is an existing subsystem. The catalog does not modify or extend it -- it reads from it.

The discovery chain:

1. `PluginManager.register_builtin_plugins()` calls `discover_all_plugins()` in `plugins/infrastructure/discovery.py`.
2. `discover_all_plugins()` scans configured directories under `src/elspeth/plugins/` (sources, transforms, transforms/azure, transforms/llm, transforms/rag, sinks) for Python files not in the `EXCLUDED_FILES` set.
3. For each file, it imports the module and finds classes that (a) inherit from `BaseSource`, `BaseTransform`, or `BaseSink`, (b) have a `name` class attribute (string, non-empty), and (c) are not abstract.
4. Discovered classes are registered with pluggy via dynamic hook implementations.
5. `PluginManager` caches them in `_sources`, `_transforms`, and `_sinks` dicts keyed by `name`.

The `CatalogServiceImpl` reads from these caches via `PluginManager.get_sources()`, `get_transforms()`, and `get_sinks()`, which return `list[type[SourceProtocol]]` etc. It also uses the by-name lookups (`get_source_by_name()`, etc.) for `get_schema()`.

Plugin descriptions are extracted from the class docstring's first non-empty line via the existing `get_plugin_description()` helper in `discovery.py`.

---

## API Response Models

Two Pydantic response models in `web/catalog/schemas.py`:

**PluginSummary** -- returned by the list endpoints. Lightweight, suitable for catalog browsing and LLM tool results.

| Field | Type | Source |
|-------|------|--------|
| `name` | `str` | Plugin class `name` attribute (e.g., `"csv"`, `"field_mapper"`, `"azure_blob"`) |
| `description` | `str` | First line of class docstring via `get_plugin_description()` |
| `plugin_type` | `str` | One of `"source"`, `"transform"`, `"sink"` |
| `config_fields` | `list[ConfigFieldSummary]` | Summarized config schema fields (see below) |

**ConfigFieldSummary** -- one entry per field in the plugin's Pydantic config model.

| Field | Type | Source |
|-------|------|--------|
| `name` | `str` | Pydantic field name |
| `type` | `str` | JSON Schema type string (e.g., `"string"`, `"integer"`, `"array"`) |
| `required` | `bool` | Whether the field has no default |
| `description` | `str \| None` | Pydantic `Field(description=...)` if present |
| `default` | `Any \| None` | Default value if one exists |

**PluginSchemaInfo** -- returned by the schema endpoint. Full detail for the composer to validate config against.

| Field | Type | Source |
|-------|------|--------|
| `name` | `str` | Plugin name |
| `plugin_type` | `str` | One of `"source"`, `"transform"`, `"sink"` |
| `description` | `str` | Full class docstring |
| `json_schema` | `dict[str, Any]` | Complete Pydantic `model_json_schema()` output |

The `json_schema` field contains the raw output of `ConfigModel.model_json_schema()`, which produces a standard JSON Schema (draft 2020-12) with `$defs` for nested models, `enum` for literals, `anyOf` for unions, and `default` annotations. No post-processing or filtering is applied. The frontend and the LLM composer receive the same schema the engine uses for validation.

### Schema Resolution

`CatalogServiceImpl` maps plugin name to config model class using `PluginConfigValidator`'s private `_get_source_config_model()`, `_get_transform_config_model()`, and `_get_sink_config_model()` methods. These methods contain the authoritative name-to-config-class mapping. To avoid duplicating this mapping, `CatalogServiceImpl` delegates to `PluginConfigValidator` for config model lookup.

For the `llm` transform, which dispatches to provider-specific config models, the schema endpoint returns the base `LLMConfig` schema (which includes the `provider` literal field). The caller can then request the provider-specific schema once the provider is known. This matches the existing validation flow.

Plugins with no config model (e.g., `null` source) return a `PluginSchemaInfo` with an empty `json_schema` (`{}`).

---

## REST API

All endpoints are read-only. All responses are JSON. All are mounted under the `/api/catalog` prefix via a FastAPI `APIRouter` in `web/catalog/routes.py`.

**GET /api/catalog/sources**

Returns `list[PluginSummary]` of all registered source plugins. Status 200.

**GET /api/catalog/transforms**

Returns `list[PluginSummary]` of all registered transform plugins. Gates are excluded. Status 200.

**GET /api/catalog/sinks**

Returns `list[PluginSummary]` of all registered sink plugins. Status 200.

**GET /api/catalog/{type}/{name}/schema**

Path parameters:
- `type`: One of `sources`, `transforms`, `sinks`.
- `name`: Plugin name (e.g., `csv`, `field_mapper`).

Returns `PluginSchemaInfo` with the full JSON schema. Status 200.

Error responses:
- 404 with `{"detail": "Unknown plugin type: {type}"}` if `type` is not one of the three valid values.
- 404 with `{"detail": "Unknown {type} plugin: {name}. Available: [...]"}` if the plugin name is not registered.

All catalog endpoints require authentication via `Depends(get_current_user)`, applied at the router-mount seam in `create_app()`. The earlier v1 position (public catalog) was predicated on the schema endpoint returning only the truncated base-class schema; once the flattener was widened to emit full Pydantic discriminated-union contracts (per-provider required fields, full docstrings), publishing that surface to anonymous callers was no longer consistent with the secured-deployment posture implied by `auth_provider=oidc|entra`. Contract pinned in `tests/unit/web/test_app.py::TestCatalogAuthGating`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/catalog/__init__.py` | Module init |
| Create | `src/elspeth/web/catalog/protocol.py` | `CatalogService` protocol (4 methods) |
| Create | `src/elspeth/web/catalog/service.py` | `CatalogServiceImpl` wrapping `PluginManager` |
| Create | `src/elspeth/web/catalog/schemas.py` | `PluginSummary`, `ConfigFieldSummary`, `PluginSchemaInfo` |
| Create | `src/elspeth/web/catalog/routes.py` | FastAPI router with 4 endpoints |
| Modify | `src/elspeth/web/app.py` | Register catalog router in app factory |
| Modify | `src/elspeth/web/dependencies.py` | Add `CatalogService` dependency provider |
| Create | `tests/unit/web/catalog/__init__.py` | Test module init |
| Create | `tests/unit/web/catalog/test_service.py` | CatalogServiceImpl tests with real `PluginManager` |
| Create | `tests/unit/web/catalog/test_routes.py` | Catalog API endpoint tests via `TestClient` |

---

## Acceptance Criteria

1. `CatalogServiceImpl` discovers all built-in plugins via `PluginManager.register_builtin_plugins()` and returns them as `PluginSummary` lists. Every plugin registered in `_sources`, `_transforms`, and `_sinks` caches appears in the corresponding list.

2. `get_schema()` returns a `PluginSchemaInfo` whose `json_schema` field matches the output of `ConfigModel.model_json_schema()` for that plugin's config class. The schema is valid JSON Schema.

3. `GET /api/catalog/sources`, `GET /api/catalog/transforms`, and `GET /api/catalog/sinks` each return 200 with a JSON array of plugin summaries. Each summary includes `name`, `description`, `plugin_type`, and `config_fields`.

4. `GET /api/catalog/{type}/{name}/schema` returns 200 with a full schema for known plugins, and 404 with a descriptive error for unknown type or name.

5. No catalog endpoint modifies state. The catalog module has no POST, PUT, PATCH, or DELETE endpoints.

6. Gates do not appear in the transform list. They are config-driven system operations, not plugins.

7. `ConfigFieldSummary` correctly reflects field types, defaults, required status, and descriptions from the Pydantic config models.

8. The LLM transform returns base `LLMConfig` schema from `get_schema("transforms", "llm")`. Provider-specific schemas are not exposed as separate catalog entries.

9. All tests use real `PluginManager` with `register_builtin_plugins()` -- no mocked plugin lists.

10. `CatalogService` is wired into the FastAPI dependency injection system and available to other modules (specifically, the Composer in Phase 4 will call `list_sources()`, `list_transforms()`, `list_sinks()`, and `get_schema()` as LLM tool implementations).
