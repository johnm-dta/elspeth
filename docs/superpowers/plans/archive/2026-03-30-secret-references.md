# Secret References Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users securely provide API keys (e.g. OpenRouter, Azure) for pipeline execution without exposing plaintext to the browser or LLM.

**Architecture:** Extend the existing `SecretLoader` protocol with a user-secret backend (Fernet-encrypted in sessions.db) and compose it with the existing `EnvSecretLoader` via `CompositeSecretLoader`. A `resolve_secret_refs()` tree-walk replaces `{"secret_ref": "NAME"}` markers in pipeline config at execution time. The contract lives in `contracts/` (L0), the tree-walk in `core/` (L1), web-specific backends + REST + UI in `web/` (L3).

**Tech Stack:** cryptography (Fernet), SQLAlchemy Core, FastAPI, Zustand, existing `SecretLoader`/`CompositeSecretLoader` from `core.security`.

---

## File Structure

### New Files

| File | Layer | Responsibility |
|------|-------|---------------|
| `src/elspeth/contracts/secrets.py` | L0 | `SecretInventoryItem` dataclass, `WebSecretResolver` protocol (list + resolve), `ResolvedSecret` with safe `__repr__` |
| `src/elspeth/core/secrets.py` | L1 | `resolve_secret_refs()` tree-walk helper, `ChainedWebSecretResolver` |
| `src/elspeth/web/secrets/__init__.py` | L3 | Package marker |
| `src/elspeth/web/secrets/user_store.py` | L3 | `UserSecretStore` — Fernet-encrypted DB backend implementing `SecretLoader` + inventory |
| `src/elspeth/web/secrets/server_store.py` | L3 | `ServerSecretStore` — curated env var inventory + resolution |
| `src/elspeth/web/secrets/service.py` | L3 | `WebSecretService` — composes user + server stores, exposes list/resolve/validate |
| `src/elspeth/web/secrets/routes.py` | L3 | REST endpoints (GET /secrets, POST /secrets, DELETE, POST validate) |
| `src/elspeth/web/secrets/schemas.py` | L3 | Pydantic request/response models |
| `src/elspeth/web/sessions/migrations/versions/003_add_user_secrets.py` | L3 | Alembic migration |
| `src/elspeth/web/frontend/src/components/settings/SecretsPanel.tsx` | L3 | Write-only secret entry + inventory display |
| `src/elspeth/web/frontend/src/stores/secretsStore.ts` | L3 | Zustand store for secret inventory |
| `tests/unit/web/secrets/test_user_store.py` | Test | Encryption round-trip, isolation |
| `tests/unit/web/secrets/test_service.py` | Test | Chained resolution, inventory merge |
| `tests/unit/web/secrets/test_routes.py` | Test | REST security (no values in responses) |
| `tests/unit/core/test_resolve_secret_refs.py` | Test | Tree-walk, nested refs, missing ref errors |

### Modified Files

| File | Change |
|------|--------|
| `src/elspeth/contracts/audit.py` | Extend `_ALLOWED_SOURCES` to include `"env"` and `"user"` |
| `src/elspeth/web/sessions/models.py` | Add `user_secrets` table |
| `src/elspeth/web/app.py` | Construct `WebSecretService`, register secrets router |
| `src/elspeth/web/config.py` | Add `server_secret_allowlist` setting |
| `src/elspeth/web/composer/tools.py` | Add 3 secret tools: `list_secret_refs`, `validate_secret_ref`, `wire_secret_ref` |
| `src/elspeth/web/execution/service.py` | Call `resolve_secret_refs()` before pipeline execution |
| `src/elspeth/web/execution/validation.py` | Validate secret refs exist before execution |
| `src/elspeth/web/frontend/src/api/client.ts` | Add secret API functions |
| `src/elspeth/web/frontend/src/types/index.ts` | Add `SecretInventoryItem` type |
| `tests/unit/web/composer/test_tools.py` | Update tool count, add secret tool tests |

---

## Task 1: Extend audit contract — allow "env" and "user" sources

This is a Phase 1 blocker per the sub-plan (section 6.4). Without this, secret resolution records from the web path would be rejected by the audit system.

**Files:**
- Modify: `src/elspeth/contracts/audit.py` (lines ~763-885)
- Test: `tests/unit/audit/test_secret_resolution.py` (find existing)

- [ ] **Step 1: Find and read the existing secret resolution tests**

Run: `grep -r "SecretResolution" tests/ --include="*.py" -l`

Read the test file to understand the existing test patterns.

- [ ] **Step 2: Write tests for the new allowed sources**

```python
# In the existing test file, add:

def test_secret_resolution_input_accepts_env_source():
    inp = SecretResolutionInput(
        env_var_name="OPENROUTER_API_KEY",
        source="env",
        vault_url=None,
        secret_name=None,
        timestamp=time.time(),
        resolution_latency_ms=0.1,
        fingerprint="a" * 64,
    )
    assert inp.source == "env"


def test_secret_resolution_input_accepts_user_source():
    inp = SecretResolutionInput(
        env_var_name="OPENROUTER_API_KEY",
        source="user",
        vault_url=None,
        secret_name=None,
        timestamp=time.time(),
        resolution_latency_ms=0.1,
        fingerprint="a" * 64,
    )
    assert inp.source == "user"


def test_secret_resolution_accepts_env_source():
    """Read-side DTO also accepts the new sources."""
    res = SecretResolution(
        resolution_id="r1",
        run_id="run1",
        timestamp=time.time(),
        env_var_name="KEY",
        source="env",
        fingerprint="a" * 64,
        vault_url=None,
        secret_name=None,
        resolution_latency_ms=None,
    )
    assert res.source == "env"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/audit/ -x -v -k "secret" --timeout=30`

Expected: FAIL — "env" not in `_ALLOWED_SOURCES`

- [ ] **Step 4: Extend `_ALLOWED_SOURCES` in both DTOs**

In `src/elspeth/contracts/audit.py`, change both `SecretResolution._ALLOWED_SOURCES` and `SecretResolutionInput._ALLOWED_SOURCES`:

```python
_ALLOWED_SOURCES: ClassVar[frozenset[str]] = frozenset({"keyvault", "env", "user"})
```

Also update the validation in `SecretResolution.__post_init__` — the keyvault-specific fields (`vault_url`, `secret_name`) should only be required when `source == "keyvault"`, not for all sources:

```python
if self.source == "keyvault":
    if not self.vault_url:
        raise AuditIntegrityError("vault_url required when source is 'keyvault'")
    if not self.secret_name:
        raise AuditIntegrityError("secret_name required when source is 'keyvault'")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/audit/ -x -v -k "secret" --timeout=30`
Also: `.venv/bin/python -m pytest tests/ -x -q --timeout=60` (full suite — audit changes are high-blast-radius)

- [ ] **Step 6: Commit**

```
feat(contracts/audit): extend SecretResolution to accept "env" and "user" sources
```

---

## Task 2: Shared contracts — SecretInventoryItem, ResolvedSecret, WebSecretResolver

**Files:**
- Create: `src/elspeth/contracts/secrets.py`
- Test: `tests/unit/contracts/test_secrets.py`

- [ ] **Step 1: Write tests for the contract types**

```python
# tests/unit/contracts/test_secrets.py
"""Tests for secret contract types — security invariants."""
import pytest
from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem


class TestResolvedSecret:
    def test_repr_does_not_contain_value(self):
        """SECURITY: __repr__ must never expose plaintext."""
        rs = ResolvedSecret(name="API_KEY", value="sk-secret-123", scope="user", fingerprint="abc123")
        repr_str = repr(rs)
        assert "sk-secret-123" not in repr_str
        assert "API_KEY" in repr_str

    def test_str_does_not_contain_value(self):
        rs = ResolvedSecret(name="API_KEY", value="sk-secret-123", scope="user", fingerprint="abc123")
        assert "sk-secret-123" not in str(rs)

    def test_fields_accessible(self):
        rs = ResolvedSecret(name="KEY", value="val", scope="server", fingerprint="fp")
        assert rs.name == "KEY"
        assert rs.value == "val"
        assert rs.scope == "server"


class TestSecretInventoryItem:
    def test_no_value_field(self):
        """Inventory items must not carry secret values."""
        item = SecretInventoryItem(name="KEY", scope="user", available=True)
        assert not hasattr(item, "value")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_secrets.py -x -v`

- [ ] **Step 3: Implement the contract types**

```python
# src/elspeth/contracts/secrets.py
"""Secret resolution contracts — shared across CLI and web.

Layer: L0 (contracts). No upward imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ResolvedSecret:
    """A resolved secret value with provenance metadata.

    The value field carries plaintext for in-process runtime use ONLY.
    It must NEVER be persisted, logged, or returned in any API response.
    """

    name: str
    value: str
    scope: str  # "user", "server", "org"
    fingerprint: str

    def __repr__(self) -> str:
        return f"ResolvedSecret(name={self.name!r}, scope={self.scope!r}, fingerprint={self.fingerprint!r})"

    def __str__(self) -> str:
        return f"ResolvedSecret({self.name}, scope={self.scope})"


@dataclass(frozen=True, slots=True)
class SecretInventoryItem:
    """Browser-safe secret metadata — no value, no masked derivative."""

    name: str
    scope: str
    available: bool
    source_kind: str = ""


@runtime_checkable
class WebSecretResolver(Protocol):
    """Protocol for web-facing secret resolution and inventory."""

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]: ...

    def has_ref(self, user_id: str, name: str) -> bool: ...

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_secrets.py -x -v`

- [ ] **Step 5: Commit**

```
feat(contracts/secrets): add ResolvedSecret, SecretInventoryItem, WebSecretResolver protocol
```

---

## Task 3: Core tree-walk helper — resolve_secret_refs()

**Files:**
- Create: `src/elspeth/core/secrets.py`
- Test: `tests/unit/core/test_resolve_secret_refs.py`

- [ ] **Step 1: Write tests for the tree-walk**

```python
# tests/unit/core/test_resolve_secret_refs.py
"""Tests for resolve_secret_refs() — the config tree-walk that replaces secret_ref markers."""
import pytest
from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem
from elspeth.core.secrets import SecretResolutionError, resolve_secret_refs


class FakeResolver:
    """Test double — resolves from a dict of known secrets."""

    def __init__(self, secrets: dict[str, str]):
        self._secrets = secrets

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        return [SecretInventoryItem(name=k, scope="test", available=True) for k in self._secrets]

    def has_ref(self, user_id: str, name: str) -> bool:
        return name in self._secrets

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None:
        if name not in self._secrets:
            return None
        return ResolvedSecret(name=name, value=self._secrets[name], scope="test", fingerprint="fp")


class TestResolveSecretRefs:
    def test_replaces_secret_ref_in_flat_dict(self):
        config = {"api_key": {"secret_ref": "MY_KEY"}, "model": "gpt-4"}
        resolver = FakeResolver({"MY_KEY": "sk-123"})
        result, resolutions = resolve_secret_refs(config, resolver, "user1")
        assert result == {"api_key": "sk-123", "model": "gpt-4"}
        assert len(resolutions) == 1
        assert resolutions[0].name == "MY_KEY"

    def test_replaces_nested_secret_ref(self):
        config = {"source": {"options": {"credentials": {"secret_ref": "CRED"}}}}
        resolver = FakeResolver({"CRED": "secret-val"})
        result, _ = resolve_secret_refs(config, resolver, "user1")
        assert result["source"]["options"]["credentials"] == "secret-val"

    def test_replaces_secret_ref_in_list(self):
        config = {"keys": [{"secret_ref": "K1"}, {"secret_ref": "K2"}]}
        resolver = FakeResolver({"K1": "v1", "K2": "v2"})
        result, resolutions = resolve_secret_refs(config, resolver, "user1")
        assert result["keys"] == ["v1", "v2"]
        assert len(resolutions) == 2

    def test_raises_on_missing_secret(self):
        config = {"api_key": {"secret_ref": "MISSING"}}
        resolver = FakeResolver({})
        with pytest.raises(SecretResolutionError, match="MISSING"):
            resolve_secret_refs(config, resolver, "user1")

    def test_collects_all_missing_secrets(self):
        config = {"a": {"secret_ref": "X"}, "b": {"secret_ref": "Y"}}
        resolver = FakeResolver({})
        with pytest.raises(SecretResolutionError) as exc_info:
            resolve_secret_refs(config, resolver, "user1")
        assert "X" in str(exc_info.value)
        assert "Y" in str(exc_info.value)

    def test_leaves_non_ref_dicts_unchanged(self):
        config = {"options": {"path": "/data/file.csv", "delimiter": ","}}
        resolver = FakeResolver({})
        result, resolutions = resolve_secret_refs(config, resolver, "user1")
        assert result == config
        assert len(resolutions) == 0

    def test_empty_config(self):
        result, resolutions = resolve_secret_refs({}, FakeResolver({}), "u")
        assert result == {}
        assert resolutions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_resolve_secret_refs.py -x -v`

- [ ] **Step 3: Implement resolve_secret_refs()**

```python
# src/elspeth/core/secrets.py
"""Secret resolution helpers — tree-walk and chained resolver.

Layer: L1 (core). Imports from L0 (contracts) only.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from elspeth.contracts.secrets import ResolvedSecret, WebSecretResolver


class SecretResolutionError(Exception):
    """Raised when one or more secret refs cannot be resolved."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(f"Cannot resolve secret references: {names}")


def resolve_secret_refs(
    config: dict[str, Any],
    resolver: WebSecretResolver,
    user_id: str,
) -> tuple[dict[str, Any], list[ResolvedSecret]]:
    """Walk a config dict tree and replace {"secret_ref": "NAME"} with resolved values.

    Returns (resolved_config, list_of_resolutions).
    Raises SecretResolutionError listing ALL missing refs (not one at a time).
    The returned config is a deep copy — the original is not mutated.
    """
    result = deepcopy(config)
    resolutions: list[ResolvedSecret] = []
    missing: list[str] = []
    _walk(result, resolver, user_id, resolutions, missing)
    if missing:
        raise SecretResolutionError(missing)
    return result, resolutions


def _is_secret_ref(value: Any) -> str | None:
    """If value is {"secret_ref": "NAME"}, return NAME. Else None."""
    if isinstance(value, dict) and len(value) == 1 and "secret_ref" in value:
        ref = value["secret_ref"]
        if isinstance(ref, str):
            return ref
    return None


def _walk(
    obj: Any,
    resolver: WebSecretResolver,
    user_id: str,
    resolutions: list[ResolvedSecret],
    missing: list[str],
) -> Any:
    """Recursively walk and replace secret refs in-place."""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            ref_name = _is_secret_ref(obj[key])
            if ref_name is not None:
                resolved = resolver.resolve(user_id, ref_name)
                if resolved is None:
                    missing.append(ref_name)
                else:
                    obj[key] = resolved.value
                    resolutions.append(resolved)
            else:
                _walk(obj[key], resolver, user_id, resolutions, missing)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            ref_name = _is_secret_ref(item)
            if ref_name is not None:
                resolved = resolver.resolve(user_id, ref_name)
                if resolved is None:
                    missing.append(ref_name)
                else:
                    obj[i] = resolved.value
                    resolutions.append(resolved)
            else:
                _walk(item, resolver, user_id, resolutions, missing)
    return obj
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/test_resolve_secret_refs.py -x -v`

- [ ] **Step 5: Commit**

```
feat(core/secrets): add resolve_secret_refs() tree-walk for config secret replacement
```

---

## Task 4: User secret store — Fernet-encrypted DB backend

**Files:**
- Modify: `src/elspeth/web/sessions/models.py` — add `user_secrets` table
- Create: `src/elspeth/web/sessions/migrations/versions/003_add_user_secrets.py`
- Create: `src/elspeth/web/secrets/__init__.py`
- Create: `src/elspeth/web/secrets/user_store.py`
- Test: `tests/unit/web/secrets/__init__.py`
- Test: `tests/unit/web/secrets/test_user_store.py`

- [ ] **Step 1: Add user_secrets table to models.py**

```python
# Add after blob_run_links_table in src/elspeth/web/sessions/models.py:
user_secrets_table = Table(
    "user_secrets",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("user_id", String, nullable=False),
    Column("encrypted_value", LargeBinary, nullable=False),
    Column("salt", LargeBinary, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    UniqueConstraint("name", "user_id", name="uq_user_secret_name_user"),
)
Index("ix_user_secrets_user_id", user_secrets_table.c.user_id)
```

Import `LargeBinary` from sqlalchemy.

- [ ] **Step 2: Create migration 003**

```python
# src/elspeth/web/sessions/migrations/versions/003_add_user_secrets.py
"""Add user_secrets table for encrypted user-scoped secret storage.

Revision ID: 003
Revises: 002
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str | Sequence[str] | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_secrets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("encrypted_value", sa.LargeBinary(), nullable=False),
        sa.Column("salt", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "user_id", name="uq_user_secret_name_user"),
    )
    op.create_index("ix_user_secrets_user_id", "user_secrets", ["user_id"])


def downgrade() -> None:
    op.drop_table("user_secrets")
```

- [ ] **Step 3: Write tests for UserSecretStore**

Test encryption round-trip, user isolation, upsert semantics, deletion, and that the store implements `SecretLoader.get_secret()`.

Key tests:
- `test_store_and_retrieve_secret` — round-trip through Fernet
- `test_different_users_isolated` — user A cannot see user B's secrets
- `test_upsert_updates_existing` — same name+user overwrites
- `test_delete_removes_secret` — deleted secret raises on get
- `test_list_returns_metadata_only` — no values in list results
- `test_get_secret_implements_secret_loader` — returns `(str, SecretRef)` tuple

- [ ] **Step 4: Implement UserSecretStore**

```python
# src/elspeth/web/secrets/user_store.py
"""Fernet-encrypted user secret store backed by sessions.db.

Implements the existing SecretLoader protocol for resolution,
plus inventory methods for the web API.
"""
from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from uuid import uuid4

from cryptography.fernet import Fernet
from sqlalchemy import Engine, delete, select

from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.core.security.secret_loader import SecretNotFoundError, SecretRef
from elspeth.web.sessions.models import user_secrets_table


def _derive_key(master_key: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet key from the master key + per-secret salt."""
    dk = hashlib.pbkdf2_hmac("sha256", master_key.encode(), salt, iterations=100_000)
    import base64
    return base64.urlsafe_b64encode(dk)


class UserSecretStore:
    """Fernet-encrypted user-scoped secret persistence.

    Thread-safe: each method opens its own connection. Safe to call
    from the worker thread via _call_async or directly.
    """

    def __init__(self, engine: Engine, master_key: str) -> None:
        self._engine = engine
        self._master_key = master_key

    def set_secret(self, user_id: str, name: str, value: str) -> None:
        """Create or update a user secret (upsert)."""
        salt = os.urandom(16)
        fernet = Fernet(_derive_key(self._master_key, salt))
        encrypted = fernet.encrypt(value.encode())
        now = datetime.now(UTC)

        with self._engine.begin() as conn:
            existing = conn.execute(
                select(user_secrets_table).where(
                    user_secrets_table.c.name == name,
                    user_secrets_table.c.user_id == user_id,
                )
            ).first()

            if existing is not None:
                conn.execute(
                    user_secrets_table.update()
                    .where(user_secrets_table.c.id == existing.id)
                    .values(encrypted_value=encrypted, salt=salt, updated_at=now)
                )
            else:
                conn.execute(
                    user_secrets_table.insert().values(
                        id=str(uuid4()),
                        name=name,
                        user_id=user_id,
                        encrypted_value=encrypted,
                        salt=salt,
                        created_at=now,
                        updated_at=now,
                    )
                )

    def get_secret(self, name: str, *, user_id: str) -> tuple[str, SecretRef]:
        """Resolve a user secret. SecretLoader-compatible return shape."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(user_secrets_table).where(
                    user_secrets_table.c.name == name,
                    user_secrets_table.c.user_id == user_id,
                )
            ).first()

        if row is None:
            raise SecretNotFoundError(name)

        fernet = Fernet(_derive_key(self._master_key, bytes(row.salt)))
        plaintext = fernet.decrypt(bytes(row.encrypted_value)).decode()

        return plaintext, SecretRef(name=name, fingerprint="", source="user")

    def delete_secret(self, user_id: str, name: str) -> bool:
        """Delete a user secret. Returns True if deleted, False if not found."""
        with self._engine.begin() as conn:
            result = conn.execute(
                delete(user_secrets_table).where(
                    user_secrets_table.c.name == name,
                    user_secrets_table.c.user_id == user_id,
                )
            )
            return result.rowcount > 0

    def list_secrets(self, user_id: str) -> list[SecretInventoryItem]:
        """List secret metadata for a user (no values)."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(
                    user_secrets_table.c.name,
                    user_secrets_table.c.updated_at,
                ).where(user_secrets_table.c.user_id == user_id)
            ).fetchall()

        return [
            SecretInventoryItem(name=row.name, scope="user", available=True, source_kind="encrypted_db")
            for row in rows
        ]
```

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

```
feat(web/secrets): add UserSecretStore — Fernet-encrypted user secret persistence
```

---

## Task 5: Server secret store + WebSecretService

**Files:**
- Create: `src/elspeth/web/secrets/server_store.py`
- Create: `src/elspeth/web/secrets/service.py`
- Modify: `src/elspeth/web/config.py` — add `server_secret_allowlist`
- Test: `tests/unit/web/secrets/test_service.py`

- [ ] **Step 1: Write tests for chained resolution**

Key tests:
- `test_user_secret_takes_priority_over_server` — user scope wins
- `test_server_fallback_when_user_missing` — falls through to env
- `test_list_refs_merges_scopes` — inventory shows both user and server secrets
- `test_list_refs_deduplicates_by_name` — user scope wins for display
- `test_resolve_returns_none_when_missing` — no exception, just None
- `test_has_ref_checks_all_scopes`

- [ ] **Step 2: Implement ServerSecretStore**

```python
# src/elspeth/web/secrets/server_store.py
"""Curated server-secret inventory backed by environment variables.

Only exposes secrets whose names are in the configured allowlist.
Does NOT dump arbitrary env vars to the browser.
"""
from __future__ import annotations

import os

from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.core.security.secret_loader import SecretNotFoundError, SecretRef


class ServerSecretStore:
    def __init__(self, allowlist: tuple[str, ...]) -> None:
        self._allowlist = allowlist

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        if name not in self._allowlist:
            raise SecretNotFoundError(name)
        value = os.environ.get(name)
        if not value:
            raise SecretNotFoundError(name)
        return value, SecretRef(name=name, fingerprint="", source="env")

    def list_secrets(self) -> list[SecretInventoryItem]:
        return [
            SecretInventoryItem(
                name=name,
                scope="server",
                available=bool(os.environ.get(name)),
                source_kind="env",
            )
            for name in self._allowlist
        ]
```

- [ ] **Step 3: Implement WebSecretService**

```python
# src/elspeth/web/secrets/service.py
"""WebSecretService — composes user + server stores behind WebSecretResolver."""
from __future__ import annotations

from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem, WebSecretResolver
from elspeth.core.security.secret_loader import SecretNotFoundError
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.user_store import UserSecretStore


class WebSecretService:
    """Chained secret resolution: user -> server.

    Implements WebSecretResolver for use by resolve_secret_refs().
    Also exposes set/delete for the REST API.
    """

    def __init__(self, user_store: UserSecretStore, server_store: ServerSecretStore) -> None:
        self._user_store = user_store
        self._server_store = server_store

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        user_items = {item.name: item for item in self._user_store.list_secrets(user_id)}
        server_items = {item.name: item for item in self._server_store.list_secrets()}
        # User scope wins for display when same name exists in both
        merged = {**server_items, **user_items}
        return sorted(merged.values(), key=lambda x: x.name)

    def has_ref(self, user_id: str, name: str) -> bool:
        return self.resolve(user_id, name) is not None

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None:
        # User scope first
        try:
            value, ref = self._user_store.get_secret(name, user_id=user_id)
            return ResolvedSecret(name=name, value=value, scope="user", fingerprint=ref.fingerprint)
        except SecretNotFoundError:
            pass
        # Server fallback
        try:
            value, ref = self._server_store.get_secret(name)
            return ResolvedSecret(name=name, value=value, scope="server", fingerprint=ref.fingerprint)
        except SecretNotFoundError:
            return None

    # Pass-through for REST API
    def set_user_secret(self, user_id: str, name: str, value: str) -> None:
        self._user_store.set_secret(user_id, name, value)

    def delete_user_secret(self, user_id: str, name: str) -> bool:
        return self._user_store.delete_secret(user_id, name)
```

- [ ] **Step 4: Add `server_secret_allowlist` to WebSettings**

In `src/elspeth/web/config.py`:
```python
server_secret_allowlist: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_API_KEY",
)
```

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

```
feat(web/secrets): add ServerSecretStore, WebSecretService with chained resolution
```

---

## Task 6: REST API + Pydantic schemas

**Files:**
- Create: `src/elspeth/web/secrets/schemas.py`
- Create: `src/elspeth/web/secrets/routes.py`
- Modify: `src/elspeth/web/app.py` — construct service, register router
- Test: `tests/unit/web/secrets/test_routes.py`

- [ ] **Step 1: Write route tests**

Key tests:
- `test_list_secrets_returns_metadata_no_values` — GET /secrets
- `test_create_secret_returns_ack_no_value` — POST /secrets → 201 with name+scope only
- `test_create_secret_value_not_in_response` — SECURITY: response body never contains value
- `test_delete_user_secret` — DELETE /secrets/{name} → 204
- `test_delete_server_secret_rejected` — server secrets can't be deleted via API
- `test_validate_existing_secret` — POST /secrets/{name}/validate → available: true
- `test_validate_missing_secret` — → available: false

- [ ] **Step 2: Implement schemas**

```python
# src/elspeth/web/secrets/schemas.py
from __future__ import annotations
from pydantic import BaseModel


class SecretInventoryResponse(BaseModel):
    name: str
    scope: str
    available: bool
    source_kind: str = ""


class CreateSecretRequest(BaseModel):
    name: str
    value: str


class CreateSecretResponse(BaseModel):
    name: str
    scope: str
    available: bool


class ValidateSecretResponse(BaseModel):
    name: str
    available: bool
```

- [ ] **Step 3: Implement routes**

4 endpoints:
- `GET /api/secrets` — list inventory (all scopes)
- `POST /api/secrets` — create user secret (write-only)
- `DELETE /api/secrets/{name}` — delete user secret
- `POST /api/secrets/{name}/validate` — check existence

All require authentication. POST /secrets clears value from response immediately.

- [ ] **Step 4: Wire into app.py**

Construct `UserSecretStore(session_engine, settings.secret_key)`, `ServerSecretStore(settings.server_secret_allowlist)`, `WebSecretService(user_store, server_store)`. Store in `app.state.secret_service`. Register router.

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

```
feat(web/secrets): add REST API — list, create (write-only), delete, validate
```

---

## Task 7: Composer tools — list_secret_refs, validate_secret_ref, wire_secret_ref

**Files:**
- Modify: `src/elspeth/web/composer/tools.py`
- Modify: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Add 3 tool definitions to `get_tool_definitions()`**

- `list_secret_refs` — discovery, no params
- `validate_secret_ref` — discovery, `{name: str}`
- `wire_secret_ref` — mutation, `{field_path: str, name: str}` — sets `{"secret_ref": name}` in source/node/output options

- [ ] **Step 2: Implement handlers**

`wire_secret_ref` is the interesting one — it takes a dot-path like `"source.options.api_key"` and sets the value to `{"secret_ref": "NAME"}` in the composition state. The handler navigates the state structure and returns a mutation result.

`list_secret_refs` and `validate_secret_ref` need the `WebSecretService` — pass it the same way as `session_engine` (via `execute_tool` kwargs).

- [ ] **Step 3: Update tool count test**

19 tools total: 5 discovery + 8 mutation + 3 blob + 3 secret.

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```
feat(web/composer): add secret reference tools — list, validate, wire
```

---

## Task 8: Execution integration — resolve secrets before pipeline run

**Files:**
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/validation.py`

- [ ] **Step 1: Add secret validation check**

In `validate_pipeline()`, after the path allowlist check: walk the composition state for `{"secret_ref": ...}` patterns and check each ref exists via `WebSecretService.has_ref()`.

- [ ] **Step 2: Add resolution in _run_pipeline()**

In `_run_pipeline()`, after generating the pipeline YAML and before `load_settings()`: call `resolve_secret_refs()` on the generated config dict. Inject resolved values into the config. Pass `SecretResolutionInput` records to the orchestrator for audit trail recording.

This runs in the worker thread — `WebSecretService.resolve()` is synchronous (UserSecretStore and ServerSecretStore are both sync), so no async bridging needed.

- [ ] **Step 3: Wire WebSecretService into ExecutionServiceImpl**

Add `secret_service: WebSecretService | None = None` to the constructor. Pass from `app.py` lifespan.

- [ ] **Step 4: Run full test suite**

- [ ] **Step 5: Commit**

```
feat(web/execution): resolve secret refs at execution time with audit provenance
```

---

## Task 9: Frontend — secrets settings panel + store

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/secretsStore.ts`
- Create: `src/elspeth/web/frontend/src/components/settings/SecretsPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`

- [ ] **Step 1: Add types and API functions**

```typescript
// types/index.ts
export interface SecretInventoryItem {
  name: string;
  scope: "user" | "server" | "org";
  available: boolean;
  source_kind: string;
}
```

API functions: `listSecrets()`, `createSecret(name, value)`, `deleteSecret(name)`, `validateSecret(name)`.

- [ ] **Step 2: Create secretsStore**

Zustand store with: `secrets: SecretInventoryItem[]`, `loadSecrets()`, `createSecret(name, value)` (clears value from store immediately after API call), `deleteSecret(name)`.

- [ ] **Step 3: Create SecretsPanel**

Write-only form: name input, password-type value input, submit button. On submit: call `createSecret`, clear the value field, refresh the inventory. Inventory list shows name + scope + available badge. Delete button on user-scoped secrets only.

**SECURITY:** The value field uses `type="password"` and is cleared to `""` after successful submission. The store never retains the value.

- [ ] **Step 4: Integrate into app layout**

Add a settings/gear icon that opens the SecretsPanel. Can be a modal or a route — depends on existing layout patterns.

- [ ] **Step 5: Run frontend tests + type check**

- [ ] **Step 6: Commit**

```
feat(web/frontend): add secrets settings panel — write-only entry with inventory display
```

---

## Task 10: Create org-scope filigree issue

- [ ] **Step 1: Create filigree issue for deferred org scope**

```
title: web/secrets — implement org-scoped secret backend when workspace/team ships
type: task
priority: 3
description: AD-8 models org scope in contracts/API now but defers storage/admin.
  WebSecretResolver protocol supports org scope. ServerSecretStore could be
  extended with an OrgSecretStore backed by a shared DB or vault.
  Blocked by workspace/team infrastructure.
labels: [cluster:secrets]
```

- [ ] **Step 2: Commit any remaining changes**

```
docs: create org-scope secret deferral ticket
```

---

## Security Checklist (Self-Review)

Before marking this plan complete, verify every path:

1. **Plaintext never in REST response** — POST /secrets response has name+scope only
2. **Plaintext never in tool result** — list_secret_refs returns names, not values
3. **Plaintext never in composition state** — wire_secret_ref writes `{"secret_ref": "NAME"}`, not the value
4. **Plaintext never persisted** — encrypted in DB, resolved only in _run_pipeline worker thread
5. **ResolvedSecret.__repr__ is safe** — verified in Task 2 tests
6. **Audit records have fingerprints, not values** — SecretResolutionInput extended in Task 1
7. **Server secrets curated** — allowlist, not arbitrary env dump
8. **Frontend clears value immediately** — password field cleared after POST succeeds
