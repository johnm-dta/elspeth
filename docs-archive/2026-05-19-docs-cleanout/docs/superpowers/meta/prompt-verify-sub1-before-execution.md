# Prompt: Verify Sub-Plan 1 (Foundation) Before Execution

## Context

Sub-Plan 1 (Foundation) is 880 lines, below the 1,000-line L3 decomposition
threshold. It will execute as written — no further decomposition. Before
execution begins, verify that the plan is internally consistent, matches the
spec, and is implementable without ambiguity.

## Source of Truth

Read these files in order:

1. **Sub-spec 1:** `docs/superpowers/specs/2026-03-28-web-ux-sub1-foundation-design.md`
2. **Sub-plan 1:** `docs/superpowers/plans/2026-03-28-web-ux-sub1-foundation.md`
3. **Seam contracts:** `docs/superpowers/specs/2026-03-28-web-ux-seam-contracts.md`
4. **Program overview:** `docs/superpowers/meta/web-ux-program.md` (for R4 security rules and seam ownership)

Also read the existing code that Sub-Plan 1 modifies:

5. `src/elspeth/cli.py` — contains `_get_plugin_manager()` and `_plugin_manager_cache` to be removed
6. `src/elspeth/cli_helpers.py` — imports `_get_plugin_manager` from `cli.py`, must switch
7. `src/elspeth/plugins/infrastructure/manager.py` — target for `get_shared_plugin_manager()`
8. `pyproject.toml` — target for `[webui]` extra and `all` extra update

## Verification Checklist

### 1. Spec ↔ Plan Consistency

For every section in the spec, verify the plan has a corresponding task that
implements it. For every task in the plan, verify the spec authorises it.

- [ ] **Plugin manager extraction:** Plan task matches spec § Pre-work. `get_shared_plugin_manager()` signature, singleton contract, `register_builtin_plugins()` call, module-level cache.
- [ ] **WebSettings:** Plan task creates all fields from spec § WebSettings table. Check the field list matches exactly — especially the R4 additions: `composer_rate_limit_per_minute: int = 10` (H8p) and the S3 annotation on `secret_key`.
- [ ] **App factory:** Plan task matches spec § FastAPI Application Factory. `create_app(settings)`, CORS middleware, `/api/health`, lifespan stub, `app.state.settings`.
- [ ] **Packaging:** Plan task adds `[webui]` extra to `pyproject.toml` with the exact dependency list and version constraints from the spec. `all` extra updated to include `elspeth[webui]`.
- [ ] **CLI entry point:** Plan task adds `elspeth web` command matching spec § CLI Entry Point. Options: `--port`, `--host`, `--auth`, `--reload`. Import guard with informative error on missing `[webui]`.
- [ ] **dependencies.py:** Plan task creates `get_settings()` dependency provider. Plan notes that service stubs are added by subsequent phases.

### 2. R4 Amendment Integration

The plan has a "Round 4 Review Amendments" appendix. Verify that the appendix
items are reflected in the plan body (not just the appendix):

- [ ] `composer_rate_limit_per_minute` field appears in the WebSettings task (not just the appendix)
- [ ] `secret_key` S3 annotation appears in the WebSettings task or app factory task
- [ ] Seam contracts reference appears in the `dependencies.py` task (lifespan construction note)

### 3. Codebase Reality Check

Verify that the plan's assumptions about existing code are correct by reading
the actual files:

- [ ] **`_get_plugin_manager()` exists in `cli.py`** — grep for it. If it's been renamed or moved since the spec was written, the plan is stale.
- [ ] **`cli_helpers.py` imports from `cli.py`** — verify the exact import statement. The plan must update this specific import.
- [ ] **`PluginManager` exists in `plugins/infrastructure/manager.py`** — verify the class exists and that `register_builtin_plugins()` is a method on it or a standalone function nearby.
- [ ] **`pyproject.toml` has an `all` extra** — verify the exact key name and current contents. The plan must add `"elspeth[webui]"` to it.
- [ ] **No existing `web/` package** — verify `src/elspeth/web/` does not already exist. If it does, the plan's "Create" file actions conflict.
- [ ] **`web` extra already exists** — the spec says the extra is named `webui` because `web` is taken. Verify that `[project.optional-dependencies]` already has a `web` key.

### 4. File Map Completeness

Compare the plan's file map against the spec's file map. Every file in the spec
must appear in the plan. Flag any files in the plan that are NOT in the spec.

Spec file map:
```
Modify: src/elspeth/plugins/infrastructure/manager.py
Modify: src/elspeth/cli_helpers.py
Modify: src/elspeth/cli.py
Modify: pyproject.toml
Create: src/elspeth/web/__init__.py
Create: src/elspeth/web/app.py
Create: src/elspeth/web/config.py
Create: src/elspeth/web/dependencies.py
Create: tests/unit/plugins/test_manager_singleton.py
Create: tests/unit/web/__init__.py
Create: tests/unit/web/test_app.py
Create: tests/unit/web/test_config.py
```

### 5. Test Coverage

Verify the plan includes tests for:

- [ ] `get_shared_plugin_manager()` returns same instance on repeated calls
- [ ] `cli_helpers.py` no longer imports from `cli.py`
- [ ] `_get_plugin_manager` and `_plugin_manager_cache` no longer exist in `cli.py`
- [ ] `WebSettings` default construction produces expected values
- [ ] Invalid `auth_provider` values are rejected
- [ ] `get_landscape_url()` returns data-dir-relative default when `landscape_url` is None
- [ ] `get_payload_store_path()` returns data-dir-relative default when `payload_store_path` is None
- [ ] `get_session_db_url()` returns data-dir-relative default when `session_db_url` is None
- [ ] `create_app()` returns a FastAPI instance with `title="ELSPETH Web"`
- [ ] `GET /api/health` returns 200 with `{"status": "ok"}`
- [ ] CORS middleware is attached with configured origins
- [ ] `[webui]` extra installs cleanly

### 6. Acceptance Criteria Traceability

The spec lists 7 acceptance criteria. Verify each one maps to at least one test
in the plan:

1. Singleton extraction works → test_manager_singleton.py
2. WebSettings validates correctly → test_config.py
3. App factory produces working FastAPI app → test_app.py
4. `[webui]` extra installs cleanly → (manual or CI check)
5. CLI entry point works → (manual or integration test)
6. Existing tests pass → `pytest tests/` regression check
7. Layer compliance → `enforce_tier_model.py` CI check

### 7. Dependency Safety

Verify the plan does not introduce:

- [ ] Upward layer imports (`web/` is L3; it must not import from `tui/` or `mcp/` or other L3 peers without justification)
- [ ] Cross-CLI dependency (web entry point must not import CLI internals; the plugin manager extraction eliminates this)
- [ ] New test dependencies (test files should use existing test infrastructure)

### 8. Execution Order

Verify the plan's tasks are ordered correctly:

- [ ] Plugin manager extraction (Phase 0) comes before web package creation (Phase 1)
- [ ] `pyproject.toml` modification comes before any test that imports web dependencies
- [ ] `cli.py` modification (add `web` command) comes after web package exists

## Deliverable

Produce a verification report with:

1. **PASS / FAIL** for each checklist item
2. **Discrepancies found** — any mismatch between spec, plan, and codebase reality
3. **Ambiguities** — anything in the plan that an implementer would need to guess about
4. **Missing items** — anything in the spec not covered by the plan
5. **Verdict:** Is Sub-Plan 1 ready for execution as-is, or does it need amendments?

If the verdict is "needs amendments," list the specific changes required and their
estimated effort.
