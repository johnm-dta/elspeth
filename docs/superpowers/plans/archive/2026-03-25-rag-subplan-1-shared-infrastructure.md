# RAG Ingestion Sub-plan 1: Shared Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the foundation types and utilities that sub-plans 2, 3, and 4 all depend on — unified collection readiness result, collection probe protocol, shared ChromaDB connection config, expression parser extension, and new error types.

**Architecture:** All new types are leaf-layer (L0 contracts or L3 infrastructure). No engine changes. The `ExpressionParser` extension adds dict-context support to the existing AST-whitelist parser. `ChromaConnectionConfig` extracts shared fields from the existing `ChromaSearchProviderConfig`. Error types follow existing plain-Exception patterns.

**Tech Stack:** Python dataclasses (frozen), Pydantic v2 BaseModel, `ast` module, `@runtime_checkable` Protocol

**Spec:** `docs/superpowers/specs/2026-03-25-rag-ingestion-pipeline-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/elspeth/contracts/probes.py` | `CollectionReadinessResult` dataclass, `CollectionProbe` protocol |
| Modify | `src/elspeth/contracts/__init__.py` | Export new types |
| Modify | `src/elspeth/contracts/errors.py` | Add `DependencyFailedError`, `CommencementGateFailedError`, `RetrievalNotReadyError` |
| Create | `src/elspeth/plugins/infrastructure/clients/retrieval/connection.py` | `ChromaConnectionConfig` shared Pydantic model |
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py` | Refactor `ChromaSearchProviderConfig` to compose `ChromaConnectionConfig` |
| Modify | `src/elspeth/core/expression_parser.py` | Extend `evaluate()` to accept plain dict contexts for gate evaluation |
| Create | `tests/unit/contracts/test_probes.py` | Tests for `CollectionReadinessResult` and `CollectionProbe` |
| Create | `tests/unit/contracts/test_new_errors.py` | Tests for new error types |
| Create | `tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py` | Tests for `ChromaConnectionConfig` |
| Modify | `tests/unit/core/test_expression_parser.py` | Tests for dict-context evaluation |

---

### Task 1: `CollectionReadinessResult` Dataclass

**Files:**
- Create: `src/elspeth/contracts/probes.py`
- Create: `tests/unit/contracts/test_probes.py`

- [ ] **Step 1: Write the test for `CollectionReadinessResult` construction and immutability**

```python
# tests/unit/contracts/test_probes.py
"""Tests for collection readiness probes."""

from __future__ import annotations

import pytest

from elspeth.contracts.probes import CollectionReadinessResult


class TestCollectionReadinessResult:
    """Tests for the unified collection readiness result type."""

    def test_construction_with_all_fields(self) -> None:
        result = CollectionReadinessResult(
            collection="science-facts",
            reachable=True,
            count=450,
            message="Collection 'science-facts' has 450 documents",
        )
        assert result.collection == "science-facts"
        assert result.reachable is True
        assert result.count == 450
        assert result.message == "Collection 'science-facts' has 450 documents"

    def test_frozen_immutability(self) -> None:
        result = CollectionReadinessResult(
            collection="test",
            reachable=True,
            count=10,
            message="ok",
        )
        with pytest.raises(AttributeError):
            result.count = 99  # type: ignore[misc]

    def test_unreachable_result(self) -> None:
        result = CollectionReadinessResult(
            collection="missing",
            reachable=False,
            count=0,
            message="Collection 'missing' not found",
        )
        assert result.reachable is False
        assert result.count == 0

    def test_empty_collection_result(self) -> None:
        result = CollectionReadinessResult(
            collection="empty",
            reachable=True,
            count=0,
            message="Collection 'empty' is empty",
        )
        assert result.reachable is True
        assert result.count == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_probes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'elspeth.contracts.probes'`

- [ ] **Step 3: Implement `CollectionReadinessResult`**

```python
# src/elspeth/contracts/probes.py
"""Collection readiness probes — protocols and result types.

Used by commencement gates (pre-flight checks) and retrieval provider
readiness contracts (transform pre-conditions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CollectionReadinessResult:
    """Result of a collection readiness check.

    All fields are scalars — no __post_init__ freeze guard needed.
    """

    collection: str
    reachable: bool
    count: int
    message: str


@runtime_checkable
class CollectionProbe(Protocol):
    """Probes a vector store collection for readiness.

    Implementations live in L3 (plugins/infrastructure).
    The protocol is L0 so L2 (engine) can depend on it.
    """

    collection_name: str

    def probe(self) -> CollectionReadinessResult: ...
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_probes.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/probes.py tests/unit/contracts/test_probes.py
git commit -m "feat: add CollectionReadinessResult and CollectionProbe protocol"
```

---

### Task 2: `CollectionProbe` Protocol Tests

**Files:**
- Modify: `tests/unit/contracts/test_probes.py`

- [ ] **Step 1: Write tests for `CollectionProbe` protocol compliance**

```python
# Append to tests/unit/contracts/test_probes.py

from elspeth.contracts.probes import CollectionProbe


class TestCollectionProbe:
    """Tests for the CollectionProbe protocol."""

    def test_compliant_implementation_passes_isinstance(self) -> None:
        class FakeProbe:
            collection_name: str = "test-collection"

            def probe(self) -> CollectionReadinessResult:
                return CollectionReadinessResult(
                    collection=self.collection_name,
                    reachable=True,
                    count=5,
                    message="ok",
                )

        probe = FakeProbe()
        assert isinstance(probe, CollectionProbe)

    def test_non_compliant_missing_probe_method(self) -> None:
        class BadProbe:
            collection_name: str = "test"

        assert not isinstance(BadProbe(), CollectionProbe)

    def test_non_compliant_missing_collection_name(self) -> None:
        class BadProbe:
            def probe(self) -> CollectionReadinessResult:
                return CollectionReadinessResult(
                    collection="x", reachable=True, count=1, message="ok"
                )

        # runtime_checkable checks callable members but may not check
        # non-callable class attributes consistently — this test documents
        # the current behaviour regardless of outcome.
        _ = isinstance(BadProbe(), CollectionProbe)
```

- [ ] **Step 2: Run to verify tests pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_probes.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_probes.py
git commit -m "test: add CollectionProbe protocol compliance tests"
```

---

### Task 3: New Error Types

**Files:**
- Modify: `src/elspeth/contracts/errors.py`
- Create: `tests/unit/contracts/test_new_errors.py`

- [ ] **Step 1: Write tests for the three new error types**

```python
# tests/unit/contracts/test_new_errors.py
"""Tests for RAG ingestion error types."""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import (
    CommencementGateFailedError,
    DependencyFailedError,
    RetrievalNotReadyError,
)


class TestDependencyFailedError:
    def test_construction_and_message(self) -> None:
        err = DependencyFailedError(
            dependency_name="index_corpus",
            run_id="abc-123",
            reason="Source file not found",
        )
        assert "index_corpus" in str(err)
        assert "abc-123" in str(err)
        assert err.dependency_name == "index_corpus"
        assert err.run_id == "abc-123"
        assert err.reason == "Source file not found"

    def test_is_exception(self) -> None:
        err = DependencyFailedError(
            dependency_name="x", run_id="y", reason="z"
        )
        assert isinstance(err, Exception)


class TestCommencementGateFailedError:
    def test_construction_and_message(self) -> None:
        snapshot = {"collections": {"test": {"count": 0, "reachable": True}}}
        err = CommencementGateFailedError(
            gate_name="corpus_ready",
            condition="collections['test']['count'] > 0",
            reason="Condition evaluated to falsy",
            context_snapshot=snapshot,
        )
        assert "corpus_ready" in str(err)
        assert err.gate_name == "corpus_ready"
        assert err.condition == "collections['test']['count'] > 0"
        assert err.context_snapshot == snapshot

    def test_is_exception(self) -> None:
        err = CommencementGateFailedError(
            gate_name="x",
            condition="True",
            reason="z",
            context_snapshot={},
        )
        assert isinstance(err, Exception)


class TestRetrievalNotReadyError:
    def test_construction_and_message(self) -> None:
        err = RetrievalNotReadyError(
            "RAG transform 'retrieve' requires a populated collection. "
            "Collection 'science-facts' is empty"
        )
        assert "science-facts" in str(err)

    def test_is_exception(self) -> None:
        err = RetrievalNotReadyError("test")
        assert isinstance(err, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_new_errors.py -v`
Expected: FAIL — `ImportError: cannot import name 'DependencyFailedError'`

- [ ] **Step 3: Add the three error types to `contracts/errors.py`**

Add at the end of the file, before any final `__all__` if present:

```python
class DependencyFailedError(Exception):
    """A pipeline dependency failed to complete successfully."""

    def __init__(
        self, *, dependency_name: str, run_id: str, reason: str
    ) -> None:
        self.dependency_name = dependency_name
        self.run_id = run_id
        self.reason = reason
        super().__init__(
            f"Dependency '{dependency_name}' failed (run_id={run_id}): {reason}"
        )


class CommencementGateFailedError(Exception):
    """A commencement gate evaluated to falsy or raised an error."""

    def __init__(
        self,
        *,
        gate_name: str,
        condition: str,
        reason: str,
        context_snapshot: Mapping[str, Any],
    ) -> None:
        self.gate_name = gate_name
        self.condition = condition
        self.reason = reason
        self.context_snapshot = context_snapshot
        super().__init__(
            f"Commencement gate '{gate_name}' failed: {reason} "
            f"(condition: {condition})"
        )


class RetrievalNotReadyError(Exception):
    """A retrieval provider's collection is empty or unreachable."""

    pass
```

Ensure `from __future__ import annotations` is at the top and `Mapping` is imported from `collections.abc` (check existing imports — it may already be imported).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_new_errors.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/errors.py src/elspeth/contracts/probes.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/errors.py tests/unit/contracts/test_new_errors.py
git commit -m "feat: add DependencyFailedError, CommencementGateFailedError, RetrievalNotReadyError"
```

---

### Task 4: `ChromaConnectionConfig` Shared Model

**Files:**
- Create: `src/elspeth/plugins/infrastructure/clients/retrieval/connection.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`
- Create: `tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py`

- [ ] **Step 1: Write tests for `ChromaConnectionConfig` validation**

```python
# tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py
"""Tests for shared ChromaDB connection configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)


class TestChromaConnectionConfig:
    def test_persistent_mode_requires_persist_directory(self) -> None:
        with pytest.raises(ValidationError, match="persist_directory"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
            )

    def test_persistent_mode_forbids_host(self) -> None:
        with pytest.raises(ValidationError, match="host"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
                persist_directory="./data",
                host="example.com",
            )

    def test_client_mode_requires_host(self) -> None:
        with pytest.raises(ValidationError, match="host"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
            )

    def test_client_mode_forbids_persist_directory(self) -> None:
        with pytest.raises(ValidationError, match="persist_directory"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="example.com",
                persist_directory="./data",
            )

    def test_client_mode_requires_https_for_remote(self) -> None:
        with pytest.raises(ValidationError, match="SSL|HTTPS|https"):
            ChromaConnectionConfig(
                collection="test",
                mode="client",
                host="remote.example.com",
                ssl=False,
            )

    def test_client_mode_allows_no_ssl_for_localhost(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="client",
            host="localhost",
            ssl=False,
        )
        assert config.host == "localhost"
        assert config.ssl is False

    def test_persistent_mode_valid(self) -> None:
        config = ChromaConnectionConfig(
            collection="science-facts",
            mode="persistent",
            persist_directory="./chroma_data",
            distance_function="cosine",
        )
        assert config.collection == "science-facts"
        assert config.mode == "persistent"
        assert config.persist_directory == "./chroma_data"

    def test_defaults(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="persistent",
            persist_directory="./data",
        )
        assert config.port == 8000
        assert config.distance_function == "cosine"
        assert config.ssl is True

    def test_frozen(self) -> None:
        config = ChromaConnectionConfig(
            collection="test",
            mode="persistent",
            persist_directory="./data",
        )
        with pytest.raises(ValidationError):
            config.collection = "other"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            ChromaConnectionConfig(
                collection="test",
                mode="persistent",
                persist_directory="./data",
                unknown_field="value",  # type: ignore[call-arg]
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ChromaConnectionConfig`**

```python
# src/elspeth/plugins/infrastructure/clients/retrieval/connection.py
"""Shared ChromaDB connection configuration.

Used by ChromaSinkConfig, ChromaSearchProviderConfig, and CollectionProbeConfig.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChromaConnectionConfig(BaseModel):
    """Shared ChromaDB connection fields with cross-field validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = Field(description="ChromaDB collection name")
    mode: Literal["persistent", "client"] = Field(
        description="Connection mode: persistent (local disk) or client (remote HTTP)"
    )
    persist_directory: str | None = Field(
        default=None,
        description="Path to ChromaDB data directory (persistent mode only)",
    )
    host: str | None = Field(
        default=None,
        description="ChromaDB server hostname (client mode only)",
    )
    port: int = Field(default=8000, description="ChromaDB server port")
    ssl: bool = Field(default=True, description="Use HTTPS for client connections")
    distance_function: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance function for collection creation",
    )

    @model_validator(mode="after")
    def validate_mode_fields(self) -> ChromaConnectionConfig:
        if self.mode == "persistent":
            if self.persist_directory is None:
                raise ValueError(
                    "persist_directory is required when mode='persistent'"
                )
            if self.host is not None:
                raise ValueError(
                    "host must not be set when mode='persistent'"
                )
        elif self.mode == "client":
            if self.host is None:
                raise ValueError("host is required when mode='client'")
            if self.persist_directory is not None:
                raise ValueError(
                    "persist_directory must not be set when mode='client'"
                )
            if not self.ssl and self.host not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(
                    f"HTTPS (ssl=True) is required for remote ChromaDB hosts, "
                    f"got host={self.host!r} with ssl=False. "
                    f"Non-SSL connections are only permitted for localhost."
                )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/infrastructure/clients/retrieval/connection.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/clients/retrieval/connection.py tests/unit/plugins/infrastructure/clients/retrieval/test_connection.py
git commit -m "feat: add ChromaConnectionConfig shared Pydantic model"
```

---

### Task 5: Refactor `ChromaSearchProviderConfig` to Compose `ChromaConnectionConfig`

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`
- Modify: `tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py`

- [ ] **Step 1: Run existing Chroma provider tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py -v`
Expected: PASS (all existing tests)

- [ ] **Step 2: Refactor `ChromaSearchProviderConfig` to compose `ChromaConnectionConfig`**

In `chroma.py`, replace the duplicated connection fields with a composed `ChromaConnectionConfig`. The existing `ChromaSearchProviderConfig` also supports `mode="ephemeral"` which `ChromaConnectionConfig` does not (the shared config is for persistent and client modes only). Keep the ephemeral mode on `ChromaSearchProviderConfig` by using a discriminated union or by keeping `mode` as a wider Literal on the provider config and only using `ChromaConnectionConfig` for the shared fields when mode is persistent or client.

**Approach:** Extract the shared fields into `ChromaConnectionConfig` usage, but keep `ChromaSearchProviderConfig` as the provider-facing config. The provider config can accept `mode: Literal["ephemeral", "persistent", "client"]` and delegate persistent/client validation to `ChromaConnectionConfig` internally. This is the minimal-disruption refactor.

Read the existing `ChromaSearchProviderConfig` carefully and preserve all existing validation behaviour. The key change is that shared field validation (mode/host/persist_directory mutual exclusion, SSL enforcement) now comes from `ChromaConnectionConfig`.

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py -v`
Expected: PASS (all existing tests still pass)

- [ ] **Step 4: Run full test suite for retrieval module**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py
git commit -m "refactor: compose ChromaConnectionConfig into ChromaSearchProviderConfig"
```

---

### Task 6: Extend `ExpressionParser` for Dict Contexts

The existing `ExpressionParser` has a two-layer security model:
1. **`_ExpressionValidator`** — AST walker that rejects forbidden node types. It enforces that subscript access (`x['key']`) is only permitted when the root name is `"row"` (via `_is_row_derived()` and `visit_Name()`). Names like `collections` or `dependency_runs` are rejected as `"Forbidden name"`.
2. **`_ExpressionEvaluator`** — evaluates the validated AST against a namespace. Already accepts `dict[str, Any] | PipelineRow`.

**Both layers must be extended.** The evaluator already handles dicts, but the validator rejects expressions before they reach the evaluator. The fix is to make the validator accept a configurable set of allowed top-level names (defaulting to `{"row"}` for backward compatibility).

**Files:**
- Modify: `src/elspeth/core/expression_parser.py`
- Modify: `tests/unit/engine/test_expression_parser.py`

- [ ] **Step 1: Run existing expression parser tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_expression_parser.py -v`
Expected: PASS (all existing tests)

- [ ] **Step 2: Write tests for dict-context evaluation**

Append a new test class to `tests/unit/engine/test_expression_parser.py`:

```python
class TestExpressionParserDictContext:
    """Tests for evaluating expressions against plain dict contexts.

    Commencement gates use ExpressionParser with allowed_names=['collections',
    'dependency_runs', 'env'] instead of the default ['row'].
    """

    def test_dict_subscript_access(self) -> None:
        parser = ExpressionParser(
            "collections['science-facts']['count'] > 0",
            allowed_names=["collections", "dependency_runs", "env"],
        )
        context = {
            "collections": {"science-facts": {"count": 450, "reachable": True}}
        }
        assert parser.evaluate(context) is True

    def test_dict_subscript_zero_count(self) -> None:
        parser = ExpressionParser(
            "collections['science-facts']['count'] > 0",
            allowed_names=["collections", "dependency_runs", "env"],
        )
        context = {
            "collections": {"science-facts": {"count": 0, "reachable": False}}
        }
        assert parser.evaluate(context) is False

    def test_nested_dict_boolean_logic(self) -> None:
        parser = ExpressionParser(
            "dependency_runs['index']['status'] == 'completed' "
            "and collections['test']['count'] > 10",
            allowed_names=["collections", "dependency_runs", "env"],
        )
        context = {
            "dependency_runs": {"index": {"status": "completed"}},
            "collections": {"test": {"count": 50, "reachable": True}},
        }
        assert parser.evaluate(context) is True

    def test_env_access(self) -> None:
        parser = ExpressionParser(
            "env['ENVIRONMENT'] == 'production'",
            allowed_names=["collections", "dependency_runs", "env"],
        )
        context = {"env": {"ENVIRONMENT": "production"}}
        assert parser.evaluate(context) is True

    def test_missing_key_raises(self) -> None:
        parser = ExpressionParser(
            "collections['missing']['count'] > 0",
            allowed_names=["collections"],
        )
        context = {"collections": {}}
        with pytest.raises(KeyError):
            parser.evaluate(context)

    def test_default_allowed_names_still_works(self) -> None:
        """Backward compat: default allowed_names=['row'] still works."""
        parser = ExpressionParser("row['x'] > 0")
        assert parser.evaluate({"x": 5}) is True

    def test_forbidden_name_rejected_even_with_custom_allowed(self) -> None:
        """Names not in allowed_names are still rejected."""
        with pytest.raises(ExpressionSecurityError):
            ExpressionParser(
                "os['path']",
                allowed_names=["collections"],
            )
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_expression_parser.py::TestExpressionParserDictContext -v`
Expected: FAIL — `ExpressionParser.__init__()` does not accept `allowed_names`

- [ ] **Step 4: Extend `ExpressionParser` and `_ExpressionValidator`**

In `src/elspeth/core/expression_parser.py`:

1. Add `allowed_names` parameter to `ExpressionParser.__init__()` (around line 575):
   ```python
   def __init__(self, expression: str, *, allowed_names: list[str] | None = None) -> None:
       self._allowed_names = frozenset(allowed_names or ["row"])
       # ... existing parse and validate logic, passing allowed_names to validator
   ```

2. Pass `allowed_names` to `_ExpressionValidator` (around line 620):
   ```python
   validator = _ExpressionValidator(allowed_names=self._allowed_names)
   ```

3. Update `_ExpressionValidator.__init__()` to accept `allowed_names`:
   ```python
   def __init__(self, *, allowed_names: frozenset[str] = frozenset({"row"})) -> None:
       self._allowed_names = allowed_names
   ```

4. Update `_ExpressionValidator.visit_Name()` (around line 146) to check `self._allowed_names` instead of hardcoded `"row"`:
   ```python
   if node.id not in self._allowed_names and node.id not in _SAFE_CONSTANTS and node.id not in _SAFE_BUILTINS:
       raise ExpressionSecurityError(f"Forbidden name: {node.id!r}")
   ```

5. Update `_is_row_derived()` (around line 156) to check against `self._allowed_names`:
   ```python
   def _is_row_derived(self, node: ast.expr) -> bool:
       if isinstance(node, ast.Name):
           return node.id in self._allowed_names
       # ... rest of existing logic
   ```

6. Update `_ExpressionEvaluator` to handle the multi-name namespace. When the context is a dict and `allowed_names` contains names other than `"row"`, the evaluator's `visit_Name()` should look up names directly in the context dict. Read the existing `_ExpressionEvaluator.visit_Name()` to understand the current binding and extend it.

- [ ] **Step 5: Run all expression parser tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_expression_parser.py -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 6: Run the property-based safety tests too**

Run: `.venv/bin/python -m pytest tests/property/engine/test_expression_safety.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/expression_parser.py tests/unit/engine/test_expression_parser.py
git commit -m "feat: extend ExpressionParser to support configurable allowed_names for dict contexts"
```

---

### Task 7: Export New Types and Final Verification

**Files:**
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/__init__.py`

- [ ] **Step 1: Add exports to `contracts/__init__.py`**

Add to the existing imports/exports:

```python
from elspeth.contracts.probes import CollectionProbe, CollectionReadinessResult
```

And add to `__all__` if it exists.

- [ ] **Step 2: Add `ChromaConnectionConfig` export to retrieval `__init__.py`**

```python
from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)
```

And add to `__all__`.

- [ ] **Step 3: Run full project test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -x -q`
Expected: PASS (no regressions across entire unit test suite)

- [ ] **Step 4: Run type checker on all modified files**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/probes.py src/elspeth/contracts/errors.py src/elspeth/plugins/infrastructure/clients/retrieval/connection.py src/elspeth/core/expression_parser.py`
Expected: PASS

- [ ] **Step 5: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/contracts/probes.py src/elspeth/contracts/errors.py src/elspeth/plugins/infrastructure/clients/retrieval/connection.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/__init__.py src/elspeth/plugins/infrastructure/clients/retrieval/__init__.py
git commit -m "feat: export CollectionReadinessResult, CollectionProbe, ChromaConnectionConfig"
```
