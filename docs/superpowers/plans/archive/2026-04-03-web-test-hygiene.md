# Web Test Hygiene — Coverage Gaps and Low-Value Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ~30 low-value tests from the web/ test suite and fill 2 critical coverage gaps (secrets/server_store.py and composer/prompts.py), improving signal-to-noise ratio.

**Architecture:** Pure test-layer changes — no production code modifications. Removals target tests that verify Python/Pydantic guarantees rather than application behavior. Gap fills add unit tests for two files with real business logic but no dedicated test coverage.

**Tech Stack:** pytest, monkeypatch (for env vars), unittest.mock (for catalog protocol stub)

**Worktree:** `/home/john/elspeth/.worktrees/web-test-hygiene`

**Run all commands from the worktree root.** Use `/home/john/elspeth/.venv/bin/python` as the Python interpreter.

---

## Task 1: Remove low-value tests from test_config.py

**Files:**
- Modify: `tests/unit/web/test_config.py:13-126` (remove `TestWebSettingsDefaults` and `TestWebSettingsCustomValues` classes)
- Modify: `tests/unit/web/test_config.py:172-178` (remove `TestWebSettingsImmutability` class)

**What to remove and why:**
- `TestWebSettingsDefaults` (17 tests, lines 13–85): Each test constructs `WebSettings()` and asserts a field equals its declared `Field(default=...)` value. This verifies Pydantic's default mechanism, not application logic.
- `TestWebSettingsCustomValues` (6 tests, lines 87–125): Each test constructs `WebSettings(field=value)` and asserts `settings.field == value`. Pure assignment passthrough — no validation, coercion, or derivation.
- `TestWebSettingsImmutability` (1 test, lines 172–178): Tests that `frozen=True` raises on assignment. Pydantic guarantees this.

**What to preserve** (do NOT touch):
- `TestWebSettingsValidation` (lines 128–170) — tests rejection of invalid values
- `TestWebSettingsDerivedAccessors` (lines 181–230) — tests `get_landscape_url()` derivation logic
- `TestSecretKeyGuard` (lines 233–255) — tests security-critical production guard
- `TestAuthFieldValidation` (lines 258–315) — tests conditional OIDC/Entra field requirements

- [ ] **Step 1: Remove the three low-value test classes**

Delete the following class blocks from `tests/unit/web/test_config.py`:

1. Lines 13–85: entire `class TestWebSettingsDefaults` (17 tests)
2. Lines 87–125: entire `class TestWebSettingsCustomValues` (6 tests)
3. Lines 172–178: entire `class TestWebSettingsImmutability` (1 test)

Also remove the `from pathlib import Path` import at line 5 ONLY IF no remaining tests use `Path`. Check first — `TestWebSettingsDerivedAccessors` uses `Path`, so **keep the import**.

The `from pydantic import ValidationError` import is used by `TestWebSettingsValidation` and `TestAuthFieldValidation`, so **keep it**.

After removal, the file should contain (in order):
1. Module docstring and imports
2. `TestWebSettingsValidation`
3. `TestWebSettingsDerivedAccessors`
4. `TestSecretKeyGuard`
5. `TestAuthFieldValidation`

- [ ] **Step 2: Run tests to verify nothing broke**

```bash
/home/john/elspeth/.venv/bin/python -m pytest tests/unit/web/test_config.py -v
```

Expected: All remaining tests pass. Count should drop from ~42 to ~18 tests in this file.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/test_config.py
git commit -m "test: remove 24 low-value config tests — Pydantic default/assignment/frozen guarantees"
```

---

## Task 2: Remove low-value tests from test_schemas.py, test_state.py, test_models.py, test_protocol.py

**Files:**
- Modify: `tests/unit/web/execution/test_schemas.py` (remove 7 constructor-passthrough tests)
- Modify: `tests/unit/web/composer/test_state.py` (remove 7 constructor-passthrough tests)
- Modify: `tests/unit/web/auth/test_models.py` (remove 3 tests, consolidate class)
- Modify: `tests/unit/web/sessions/test_protocol.py` (remove 2 tests)

### 2a: test_schemas.py (execution)

**Remove these tests:**
- `TestValidationResult::test_valid_result` (lines 20–49) — constructs with valid data, reads fields back. No validation tested.
- `TestRunEvent::test_progress_event` (lines 126–134) — pure constructor passthrough
- `TestRunEvent::test_completed_event` (lines 136–150) — pure constructor passthrough
- `TestRunEvent::test_error_event` (lines 152–160) — pure constructor passthrough
- `TestRunStatusResponse::test_pending_status` (lines 174–186) — pure constructor passthrough
- `TestRunStatusResponse::test_completed_status` (lines 188–200) — pure constructor passthrough
- `TestRunStatusResponse::test_failed_status_has_error` (lines 202–214) — pure constructor passthrough

**Preserve:**
- `TestValidationResult::test_invalid_result_with_attributed_error` — tests domain semantics (error attribution)
- `TestValidationResult::test_structural_error_has_null_component` — tests null component semantics
- `TestValidationResult::test_skipped_check_recorded` — tests cascade skip logic
- `TestRunEvent::test_invalid_event_type_rejected` — tests Literal validation (real constraint)

After removal, the `TestRunStatusResponse` class will be empty — delete the entire class.

- [ ] **Step 1: Remove 7 constructor-passthrough tests from test_schemas.py**

Remove the test methods and empty classes listed above. The file should retain:
1. `TestValidationResult` with 3 tests: `test_invalid_result_with_attributed_error`, `test_structural_error_has_null_component`, `test_skipped_check_recorded`
2. `TestRunEvent` with 1 test: `test_invalid_event_type_rejected`
3. No `TestRunStatusResponse` class (delete entirely)

Clean up unused imports — after removal, check if `RunStatusResponse` is still referenced. If not, remove it from the import block.

### 2b: test_state.py (composer)

**Remove these tests (all constructor passthrough):**
- `TestSourceSpec::test_create` (lines 22–31)
- `TestNodeSpec::test_create_transform` (lines 115–120+)
- `TestNodeSpec::test_create_gate` (the second pure constructor test)
- `TestEdgeSpec::test_create`
- `TestOutputSpec::test_create`
- `TestPipelineMetadata::test_defaults`
- `TestPipelineMetadata::test_custom`

**Preserve:**
- All `test_frozen` tests (verify `frozen=True` on data that matters)
- All `test_options_deep_frozen` and `test_*_nested_frozen` tests (deep_freeze contract)
- All `test_from_dict_round_trip` tests (serialization correctness)
- All `TestCompositionState` tests (business logic)

- [ ] **Step 2: Remove 7 constructor-passthrough tests from test_state.py**

Remove the test methods listed above. Each test class should retain its `test_frozen`, `test_options_deep_frozen`, and `test_from_dict_round_trip` methods.

### 2c: test_models.py (auth)

**Remove these tests:**
- `TestUserIdentity::test_construction` (lines 13–16) — trivial constructor passthrough
- `TestUserIdentity::test_slots` (lines 23–25) — testing `@dataclass` decorator guarantee
- `TestUserProfile::test_groups_is_tuple_not_list` (lines 63–70) — type annotation guarantee

**Preserve:**
- `TestUserIdentity::test_frozen_immutability` — verifies frozen contract
- `TestUserProfile::test_construction_all_fields` — documents the full interface (useful as reference)
- `TestUserProfile::test_defaults` — documents optional field defaults
- `TestUserProfile::test_frozen_immutability` — verifies frozen contract
- All `TestAuthenticationError` tests — tests real logic (default message, custom message)

After removal, `TestUserIdentity` will have only `test_frozen_immutability`. That's fine — a 1-test class for a 1-behavior type.

- [ ] **Step 3: Remove 3 low-value tests from test_models.py**

### 2d: test_protocol.py (sessions)

**Remove these tests:**
- `TestRunAlreadyActiveError::test_is_exception` (line 133) — `isinstance(err, Exception)` is a Python guarantee
- `TestSessionServiceProtocol::test_is_runtime_checkable` (line 138–139) — tests `@runtime_checkable` decorator

**Preserve:**
- `TestRunAlreadyActiveError::test_construction_and_message` — tests actual message formatting logic
- All frozen/deep-freeze tests in this file

After removal, `TestSessionServiceProtocol` class will be empty — delete the entire class.

- [ ] **Step 4: Remove 2 low-value tests from test_protocol.py**

- [ ] **Step 5: Run all web tests to verify nothing broke**

```bash
/home/john/elspeth/.venv/bin/python -m pytest tests/unit/web/ -q --tb=short
```

Expected: All remaining tests pass. Total count should drop by ~19 (from 808 to ~789).

- [ ] **Step 6: Commit**

```bash
git add tests/unit/web/execution/test_schemas.py tests/unit/web/composer/test_state.py tests/unit/web/auth/test_models.py tests/unit/web/sessions/test_protocol.py
git commit -m "test: remove 19 low-value tests — constructor passthroughs, isinstance checks, decorator guarantees"
```

---

## Task 3: Add tests for secrets/server_store.py

**Files:**
- Create: `tests/unit/web/secrets/test_server_store.py`

This file is 55 LOC but security-critical — it controls the env-var allowlist enforcement boundary. It imports `_compute_fingerprint` from `user_store.py` which requires `ELSPETH_FINGERPRINT_KEY` in the environment.

**Key behaviors to test:**
1. `has_secret()` returns `True` only for allowlisted names with set env vars
2. `has_secret()` returns `False` for non-allowlisted names (even if env var exists)
3. `has_secret()` returns `False` for allowlisted names with unset env vars
4. `get_secret()` returns `(value, SecretRef)` for allowlisted, set env vars
5. `get_secret()` raises `SecretNotFoundError` for non-allowlisted names
6. `get_secret()` raises `SecretNotFoundError` for unset env vars
7. `list_secrets()` returns metadata for all allowlisted names with correct `available` flags
8. Empty allowlist means nothing is accessible

- [ ] **Step 1: Write the test file**

Create `tests/unit/web/secrets/test_server_store.py`:

```python
"""Tests for ServerSecretStore — env-var allowlist enforcement.

Verifies:
- Allowlist boundary: non-allowlisted names always rejected
- Env-var presence: allowlisted but unset names raise SecretNotFoundError
- get_secret returns (value, SecretRef) with correct fingerprint source
- list_secrets exposes metadata only, never values
- Empty allowlist blocks everything
"""

from __future__ import annotations

import os

import pytest

from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.core.security.secret_loader import SecretNotFoundError, SecretRef
from elspeth.web.secrets.server_store import ServerSecretStore


@pytest.fixture()
def _fingerprint_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ELSPETH_FINGERPRINT_KEY is set for _compute_fingerprint."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fp-key")


@pytest.fixture()
def store(_fingerprint_key: None) -> ServerSecretStore:
    """Store with a two-item allowlist."""
    return ServerSecretStore(allowlist=("ALLOWED_KEY_A", "ALLOWED_KEY_B"))


@pytest.fixture()
def empty_store(_fingerprint_key: None) -> ServerSecretStore:
    """Store with an empty allowlist."""
    return ServerSecretStore(allowlist=())


class TestHasSecret:
    """Allowlist + env-var presence gate."""

    def test_allowlisted_and_set_returns_true(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "some-value")
        assert store.has_secret("ALLOWED_KEY_A") is True

    def test_allowlisted_but_unset_returns_false(self, store: ServerSecretStore) -> None:
        # Ensure the env var is NOT set
        os.environ.pop("ALLOWED_KEY_A", None)
        assert store.has_secret("ALLOWED_KEY_A") is False

    def test_allowlisted_but_empty_returns_false(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "")
        assert store.has_secret("ALLOWED_KEY_A") is False

    def test_not_allowlisted_returns_false_even_if_set(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NOT_ALLOWED", "secret-value")
        assert store.has_secret("NOT_ALLOWED") is False

    def test_empty_allowlist_blocks_everything(
        self, empty_store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "value")
        assert empty_store.has_secret("ALLOWED_KEY_A") is False


class TestGetSecret:
    """Value retrieval with allowlist enforcement."""

    def test_returns_value_and_ref(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "my-secret-value")
        value, ref = store.get_secret("ALLOWED_KEY_A")
        assert value == "my-secret-value"
        assert isinstance(ref, SecretRef)
        assert ref.name == "ALLOWED_KEY_A"
        assert ref.source == "env"
        assert len(ref.fingerprint) == 64  # HMAC-SHA256 hex digest

    def test_not_allowlisted_raises(self, store: ServerSecretStore) -> None:
        with pytest.raises(SecretNotFoundError):
            store.get_secret("NOT_ALLOWED")

    def test_allowlisted_but_unset_raises(self, store: ServerSecretStore) -> None:
        os.environ.pop("ALLOWED_KEY_B", None)
        with pytest.raises(SecretNotFoundError):
            store.get_secret("ALLOWED_KEY_B")

    def test_allowlisted_but_empty_raises(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "")
        with pytest.raises(SecretNotFoundError):
            store.get_secret("ALLOWED_KEY_A")

    def test_fingerprint_is_deterministic(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "same-value")
        _, ref1 = store.get_secret("ALLOWED_KEY_A")
        _, ref2 = store.get_secret("ALLOWED_KEY_A")
        assert ref1.fingerprint == ref2.fingerprint

    def test_different_values_different_fingerprints(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "value-one")
        _, ref1 = store.get_secret("ALLOWED_KEY_A")
        monkeypatch.setenv("ALLOWED_KEY_A", "value-two")
        _, ref2 = store.get_secret("ALLOWED_KEY_A")
        assert ref1.fingerprint != ref2.fingerprint


class TestListSecrets:
    """Inventory metadata without exposing values."""

    def test_lists_all_allowlisted_names(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "val-a")
        os.environ.pop("ALLOWED_KEY_B", None)
        items = store.list_secrets()
        assert len(items) == 2
        by_name = {item.name: item for item in items}
        assert by_name["ALLOWED_KEY_A"].available is True
        assert by_name["ALLOWED_KEY_B"].available is False

    def test_inventory_items_have_correct_fields(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "val")
        items = store.list_secrets()
        item = next(i for i in items if i.name == "ALLOWED_KEY_A")
        assert isinstance(item, SecretInventoryItem)
        assert item.scope == "server"
        assert item.source_kind == "env"

    def test_empty_allowlist_returns_empty(self, empty_store: ServerSecretStore) -> None:
        assert empty_store.list_secrets() == []

    def test_values_never_exposed(
        self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SecretInventoryItem has no value field — verify structurally."""
        monkeypatch.setenv("ALLOWED_KEY_A", "super-secret")
        items = store.list_secrets()
        item = items[0]
        assert not hasattr(item, "value")
```

- [ ] **Step 2: Run the new tests**

```bash
/home/john/elspeth/.venv/bin/python -m pytest tests/unit/web/secrets/test_server_store.py -v
```

Expected: All 14 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/secrets/test_server_store.py
git commit -m "test: add ServerSecretStore unit tests — allowlist enforcement, env-var boundary, fingerprint audit"
```

---

## Task 4: Add tests for composer/prompts.py

**Files:**
- Create: `tests/unit/web/composer/test_prompts.py`

This file builds LLM messages for every composition request. The critical invariant is documented in the source: `build_messages()` must return a NEW list on every call to prevent cross-turn contamination in the tool-use loop.

**Key behaviors to test:**
1. `build_messages()` returns a new list each call (identity check)
2. Message ordering: system → chat history → user
3. System message contains `SYSTEM_PROMPT` content + injected context
4. Context string includes serialized state + plugin catalog names
5. Empty chat history produces system + user only (no empty list injection)
6. `build_context_string()` includes validation summary

**Stub required:** `CatalogService` is a protocol — we need a minimal conforming stub.

- [ ] **Step 1: Write the test file**

Create `tests/unit/web/composer/test_prompts.py`:

```python
"""Tests for LLM message construction — build_messages and build_context_string.

Verifies:
- build_messages returns a NEW list on every call (cross-turn contamination guard)
- Message ordering: system → chat history → user message
- System message injects pipeline state and plugin catalog
- Empty chat history handled correctly
- Context string includes validation summary
"""

from __future__ import annotations

import json
from typing import Any

from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.prompts import SYSTEM_PROMPT, build_context_string, build_messages
from elspeth.web.composer.state import CompositionState


class StubCatalog:
    """Minimal CatalogService conforming to the protocol."""

    def list_sources(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="csv",
                description="CSV source",
                plugin_type="source",
                config_fields=[],
            )
        ]

    def list_transforms(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="uppercase",
                description="Uppercase transform",
                plugin_type="transform",
                config_fields=[],
            )
        ]

    def list_sinks(self) -> list[PluginSummary]:
        return [
            PluginSummary(
                name="csv",
                description="CSV sink",
                plugin_type="sink",
                config_fields=[],
            )
        ]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        raise ValueError(f"Not implemented for stub: {plugin_type}/{name}")


def _empty_state() -> CompositionState:
    """A minimal empty CompositionState for testing."""
    return CompositionState.from_dict(
        {
            "source": None,
            "nodes": [],
            "edges": [],
            "outputs": [],
            "metadata": {"name": "Test Pipeline", "description": ""},
            "version": 1,
        }
    )


class TestBuildMessages:
    """Message list construction and isolation."""

    def test_returns_new_list_each_call(self) -> None:
        """Critical: each call returns a distinct list object to prevent cross-turn contamination."""
        state = _empty_state()
        catalog = StubCatalog()
        history: list[dict[str, Any]] = []

        list1 = build_messages(history, state, "Hello", catalog)
        list2 = build_messages(history, state, "Hello", catalog)
        assert list1 is not list2

    def test_mutating_returned_list_does_not_affect_next_call(self) -> None:
        """Appending to a returned list must not leak into subsequent calls."""
        state = _empty_state()
        catalog = StubCatalog()

        list1 = build_messages([], state, "Hello", catalog)
        list1.append({"role": "assistant", "content": "I was injected"})

        list2 = build_messages([], state, "Hello", catalog)
        roles = [m["role"] for m in list2]
        assert "assistant" not in roles

    def test_message_ordering_system_history_user(self) -> None:
        """Messages must be: system, then history, then user."""
        state = _empty_state()
        catalog = StubCatalog()
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        messages = build_messages(history, state, "new question", catalog)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "previous answer"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "new question"

    def test_empty_history_produces_system_and_user_only(self) -> None:
        state = _empty_state()
        catalog = StubCatalog()

        messages = build_messages([], state, "my question", catalog)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my question"

    def test_system_message_contains_prompt_and_context(self) -> None:
        state = _empty_state()
        catalog = StubCatalog()

        messages = build_messages([], state, "test", catalog)
        system_content = messages[0]["content"]

        # Must contain the static system prompt
        assert SYSTEM_PROMPT in system_content
        # Must contain injected context with plugin names
        assert "csv" in system_content
        assert "uppercase" in system_content


class TestBuildContextString:
    """Context injection into the system prompt."""

    def test_contains_state_and_plugins(self) -> None:
        state = _empty_state()
        catalog = StubCatalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])  # Skip header line

        assert "current_state" in parsed
        assert "available_plugins" in parsed
        plugins = parsed["available_plugins"]
        assert "csv" in plugins["sources"]
        assert "uppercase" in plugins["transforms"]
        assert "csv" in plugins["sinks"]

    def test_includes_validation_summary(self) -> None:
        state = _empty_state()
        catalog = StubCatalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])

        validation = parsed["current_state"]["validation"]
        assert "is_valid" in validation
        assert "errors" in validation

    def test_metadata_included(self) -> None:
        state = _empty_state()
        catalog = StubCatalog()

        context = build_context_string(state, catalog)
        parsed = json.loads(context.split("\n", 1)[1])

        assert parsed["current_state"]["metadata"]["name"] == "Test Pipeline"
```

- [ ] **Step 2: Run the new tests**

```bash
/home/john/elspeth/.venv/bin/python -m pytest tests/unit/web/composer/test_prompts.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_prompts.py
git commit -m "test: add composer/prompts unit tests — message isolation, ordering, context injection"
```
