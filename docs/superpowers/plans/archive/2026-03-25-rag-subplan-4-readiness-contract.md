# RAG Ingestion Sub-plan 4: Readiness Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a readiness check to the RAG retrieval transform that refuses to start against an empty or missing collection. Implemented via `check_readiness()` on the `RetrievalProvider` protocol, with implementations on both `ChromaSearchProvider` and `AzureSearchProvider`.

**Architecture:** Extend the existing `RetrievalProvider` protocol (L3) with `check_readiness()` returning `CollectionReadinessResult` (L0, from sub-plan 1). The RAG transform calls it in `on_start()` after provider construction. Single-attempt, no retry — transient failures crash the pipeline startup.

**Tech Stack:** `@runtime_checkable` Protocol, chromadb, raw httpx (for Azure count probe — NOT AuditedHTTPClient, see Issue 3 notes below)

**Spec:** `docs/superpowers/specs/2026-03-25-rag-ingestion-pipeline-design.md` (Component 4)

**Depends on:** Sub-plan 1 (shared infrastructure) must be merged first.

**Risk:** MEDIUM-LOW — small scope but modifies a `@runtime_checkable` Protocol. All existing test doubles and mocks must be updated in the same commit.

---

## Review Notes from Sub-plan 1 (read before implementing)

The following changes were made during sub-plan 1's 4-agent review that affect this plan:

1. **`RetrievalNotReadyError` needs structured fields.** The error class was implemented as a bare `pass` class in sub-plan 1 (only a message string). The 4-agent review flagged this as an "Eroding Goals" archetype — `DependencyFailedError` and `CommencementGateFailedError` both have structured keyword fields, but `RetrievalNotReadyError` does not. When implementing the readiness contract, **add structured fields** to `RetrievalNotReadyError`: `collection_name: str`, `count: int | None`, `reachable: bool`. This aligns with `CollectionReadinessResult` and enables structured error queries from the Landscape.

2. **`CollectionProbe` protocol attribute enforcement.** Python 3.12+ `runtime_checkable` protocols DO check non-callable attributes at `isinstance()` time. This means any implementation of `CollectionProbe` MUST have a `collection_name` instance attribute — class-level annotations alone are not sufficient. Tests confirm this. When implementing `check_readiness()` on `ChromaSearchProvider` and `AzureSearchProvider`, ensure `self.collection_name = config.collection` is set in `__init__`.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/base.py` | Add `check_readiness()` to `RetrievalProvider` protocol |
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py` | Implement `check_readiness()` on `ChromaSearchProvider` |
| Modify | `src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py` | Implement `check_readiness()` on `AzureSearchProvider` |
| Modify | `src/elspeth/plugins/transforms/rag/transform.py` | Add readiness check in `on_start()` |
| Modify | `tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py` | Add `check_readiness()` tests |
| Modify | `tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py` | Add `check_readiness()` tests |
| Modify | `tests/unit/plugins/transforms/rag/test_transform.py` | Add readiness contract tests |

---

### Task 1: Extend `RetrievalProvider` Protocol

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/base.py`
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/__init__.py`

- [ ] **Step 1: Run existing retrieval tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/ -v`
Expected: PASS

- [ ] **Step 2: Add `check_readiness()` to the protocol**

In `src/elspeth/plugins/infrastructure/clients/retrieval/base.py`, add the import and method to the `RetrievalProvider` protocol (around line 28):

```python
from elspeth.contracts.probes import CollectionReadinessResult

@runtime_checkable
class RetrievalProvider(Protocol):
    """Search backend interface for RAG retrieval."""

    def search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        *,
        state_id: str,
        token_id: str | None,
    ) -> list[RetrievalChunk]:
        ...

    def check_readiness(self) -> CollectionReadinessResult:
        """Check that the target collection exists and has documents.

        Single-attempt, no retry. Called during on_start() — transient
        failures crash the pipeline startup.
        """
        ...

    def close(self) -> None:
        ...
```

- [ ] **Step 3: Add `CollectionReadinessResult` to exports**

In `__init__.py`, add:

```python
from elspeth.contracts.probes import CollectionReadinessResult
```

And add to `__all__`.

- [ ] **Step 4: Run existing tests — they will fail because providers don't implement `check_readiness()` yet**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/ -v`
Expected: May still pass (runtime_checkable only checks method names exist, and existing providers don't have it yet — but tests that call the method will fail). Note which tests fail.

- [ ] **Step 5: Commit (protocol change only)**

```bash
git add src/elspeth/plugins/infrastructure/clients/retrieval/base.py src/elspeth/plugins/infrastructure/clients/retrieval/__init__.py
git commit -m "feat: add check_readiness() to RetrievalProvider protocol"
```

---

### Task 2: `ChromaSearchProvider.check_readiness()`

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`
- Modify: `tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py`

- [ ] **Step 1: Write tests for Chroma readiness check**

Append to `tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py`:

```python
from elspeth.contracts.probes import CollectionReadinessResult


class TestChromaSearchProviderReadiness:
    """Tests for ChromaSearchProvider.check_readiness()."""

    def test_collection_with_documents_is_ready(self) -> None:
        """Collection exists and has documents."""
        provider = self._make_provider()  # Use existing test fixture pattern
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        provider._collection = mock_collection

        result = provider.check_readiness()

        assert isinstance(result, CollectionReadinessResult)
        assert result.reachable is True
        assert result.count == 10
        assert "10 documents" in result.message

    def test_empty_collection_is_not_ready(self) -> None:
        """Collection exists but is empty."""
        provider = self._make_provider()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        provider._collection = mock_collection

        result = provider.check_readiness()

        assert result.reachable is True
        assert result.count == 0
        assert "empty" in result.message

    def test_connection_error(self) -> None:
        """ChromaDB count() fails — reports unreachable with error details."""
        provider = self._make_provider()
        mock_collection = MagicMock()
        mock_collection.count.side_effect = Exception("Connection refused")
        provider._collection = mock_collection

        result = provider.check_readiness()

        assert result.reachable is False
        assert result.count == 0
        assert "Connection refused" in result.message
```

**Note:** Read the existing test file to understand the fixture pattern (`_make_provider` or similar). Adapt the test setup to match.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py::TestChromaSearchProviderReadiness -v`
Expected: FAIL — `check_readiness()` not implemented

- [ ] **Step 3: Implement `check_readiness()` on `ChromaSearchProvider`**

In `src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py`, add after `close()` (around line 255):

```python
    def check_readiness(self) -> CollectionReadinessResult:
        """Check that the ChromaDB collection exists and has documents.

        Called during on_start() AFTER provider construction. self._collection
        is always set by __init__ (which calls get_or_create_collection).
        If __init__ fails, the provider doesn't exist and this method is
        never called. No need to guard against _collection being None.
        """
        from elspeth.contracts.probes import CollectionReadinessResult

        collection_name = self._config.collection

        try:
            count = self._collection.count()
            return CollectionReadinessResult(
                collection=collection_name,
                reachable=True,
                count=count,
                message=(
                    f"Collection '{collection_name}' has {count} documents"
                    if count > 0
                    else f"Collection '{collection_name}' is empty"
                ),
            )
        except Exception as exc:
            return CollectionReadinessResult(
                collection=collection_name,
                reachable=False,
                count=0,
                message=f"Collection '{collection_name}' unreachable: {type(exc).__name__}: {exc}",
            )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py
git commit -m "feat: implement check_readiness() on ChromaSearchProvider"
```

---

### Task 3: `AzureSearchProvider.check_readiness()`

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py`
- Modify: `tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py`

- [ ] **Step 1: Write tests for Azure readiness check**

Append to `tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py`:

```python
from elspeth.contracts.probes import CollectionReadinessResult


class TestAzureSearchProviderReadiness:
    """Tests for AzureSearchProvider.check_readiness()."""

    def test_index_with_documents_is_ready(self) -> None:
        provider = self._make_provider()  # Use existing fixture pattern
        # Mock the HTTP call for document count
        # Azure Search count endpoint: GET /indexes/{index}/docs/$count
        # The provider uses AuditedHTTPClient — mock at that level
        # Read the existing test patterns for HTTP mocking
        pass  # Implement based on existing test patterns

    def test_empty_index_is_not_ready(self) -> None:
        pass  # Implement — count returns 0

    def test_index_not_found_404(self) -> None:
        pass  # Implement — HTTP 404 response

    def test_connection_error(self) -> None:
        pass  # Implement — ConnectionError or TimeoutError
```

**Note:** Read `tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py` carefully. The Azure provider uses `AuditedHTTPClient` with mocked HTTP responses. Match the existing HTTP mocking pattern exactly (likely `httpx` responses or `responses` library). The test bodies above are placeholders — fill in the actual assertions based on the existing pattern.

- [ ] **Step 2: Implement `check_readiness()` on `AzureSearchProvider`**

In `azure_search.py`, add after `close()` (around line 300):

```python
    def check_readiness(self) -> CollectionReadinessResult:
        """Check that the Azure Search index exists and has documents.

        Uses raw httpx (NOT AuditedHTTPClient) because this is a startup probe,
        not a row-level operation. There is no state_id or token_id available
        during on_start(). AuditedHTTPClient requires per-row scoping that
        doesn't exist at this lifecycle phase.
        """
        from elspeth.contracts.probes import CollectionReadinessResult

        index_name = self._config.index  # NOTE: field is 'index', not 'index_name'

        try:
            import httpx

            count_url = (
                f"{self._config.endpoint}/indexes/{index_name}"
                f"/docs/$count?api-version={self._config.api_version}"
            )
            headers = {}
            if self._config.api_key:
                headers["api-key"] = self._config.api_key
            # For managed identity, use azure-identity — but for a simple
            # readiness probe, api-key auth is sufficient. Read the existing
            # provider's auth setup to match.

            response = httpx.get(count_url, headers=headers, timeout=10.0)

            if response.status_code == 404:
                return CollectionReadinessResult(
                    collection=index_name,
                    reachable=True,
                    count=0,
                    message=f"Index '{index_name}' not found",
                )

            response.raise_for_status()
            count = int(response.text)

            return CollectionReadinessResult(
                collection=index_name,
                reachable=True,
                count=count,
                message=(
                    f"Index '{index_name}' has {count} documents"
                    if count > 0
                    else f"Index '{index_name}' is empty"
                ),
            )
        except Exception as exc:
            return CollectionReadinessResult(
                collection=index_name,
                reachable=False,
                count=0,
                message=f"Index '{index_name}' unreachable: {type(exc).__name__}: {exc}",
            )
```

**Design note:** `check_readiness()` uses raw `httpx` instead of `AuditedHTTPClient` because this is a startup probe, not a row-level pipeline operation. `AuditedHTTPClient` requires `recorder`, `state_id`, `run_id`, `telemetry_emit`, and `token_id` — none of which exist during `on_start()`. The raw HTTP call is a pre-flight check, not an audited operation. The commencement gate's `context_snapshot` records what the probe found; the HTTP call itself doesn't need per-row audit correlation.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py
git commit -m "feat: implement check_readiness() on AzureSearchProvider"
```

---

### Task 4: RAG Transform Readiness Guard

**Files:**
- Modify: `src/elspeth/plugins/transforms/rag/transform.py`
- Modify: `tests/unit/plugins/transforms/rag/test_transform.py`

- [ ] **Step 1: Write tests for readiness guard**

Append to `tests/unit/plugins/transforms/rag/test_transform.py`:

```python
from elspeth.contracts.errors import RetrievalNotReadyError
from elspeth.contracts.probes import CollectionReadinessResult


class TestRAGTransformReadinessGuard:
    """Tests for the readiness check in on_start()."""

    def test_populated_collection_passes(self) -> None:
        """on_start() succeeds when collection has documents."""
        transform = self._make_transform()  # Use existing fixture
        mock_provider = MagicMock()
        mock_provider.check_readiness.return_value = CollectionReadinessResult(
            collection="test", reachable=True, count=10,
            message="Collection 'test' has 10 documents",
        )

        # Patch provider construction to return our mock
        with patch.object(transform, "_provider", mock_provider):
            # Simulate post-on_start state where provider is set
            # Read existing test patterns for on_start testing
            pass

    def test_empty_collection_raises(self) -> None:
        """on_start() raises RetrievalNotReadyError for empty collection."""
        transform = self._make_transform()  # Use existing fixture
        mock_provider = MagicMock()
        mock_provider.check_readiness.return_value = CollectionReadinessResult(
            collection="test", reachable=True, count=0,
            message="Collection 'test' is empty",
        )

        # Patch the PROVIDERS registry to return our mock provider
        # Read existing test patterns for how on_start is tested
        with pytest.raises(RetrievalNotReadyError, match="populated collection"):
            pass  # Call on_start with the mocked provider

    def test_unreachable_collection_raises(self) -> None:
        """on_start() raises RetrievalNotReadyError for unreachable collection."""
        transform = self._make_transform()
        mock_provider = MagicMock()
        mock_provider.check_readiness.return_value = CollectionReadinessResult(
            collection="test", reachable=False, count=0,
            message="Collection 'test' unreachable",
        )

        with pytest.raises(RetrievalNotReadyError, match="populated collection"):
            pass  # Call on_start with the mocked provider
```

**Test wiring guidance:** The readiness check fires DURING `on_start()`, after provider construction. You cannot call `on_start()` and then replace `_provider` — the check has already run. Instead, mock the `PROVIDERS` registry so the factory returns a mock provider with `check_readiness` pre-configured:

```python
# Example wiring pattern:
mock_provider = MagicMock()
mock_provider.check_readiness.return_value = CollectionReadinessResult(
    collection="test", reachable=True, count=0,
    message="Collection 'test' is empty",
)

mock_factory = MagicMock(return_value=mock_provider)
mock_config_cls = MagicMock(return_value=MagicMock())

with patch.dict(
    "elspeth.plugins.transforms.rag.transform.PROVIDERS",
    {"chroma": (mock_config_cls, mock_factory)},
):
    with pytest.raises(RetrievalNotReadyError):
        transform.on_start(mock_ctx)
```

Read `tests/unit/plugins/transforms/rag/test_transform.py` for the existing mock context and config patterns. The key is patching `PROVIDERS` before calling `on_start()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/rag/test_transform.py::TestRAGTransformReadinessGuard -v`
Expected: FAIL — readiness guard not implemented

- [ ] **Step 3: Add readiness check to `on_start()`**

In `src/elspeth/plugins/transforms/rag/transform.py`, after the provider construction (line 130), add:

```python
        # Readiness check — refuse to start against empty/missing collection
        readiness = self._provider.check_readiness()
        if readiness.count == 0:
            from elspeth.contracts.errors import RetrievalNotReadyError

            raise RetrievalNotReadyError(
                collection=readiness.collection,
                reason=readiness.message,
            )
```

**IMPORTANT:** `RetrievalNotReadyError` takes keyword-only args `(*, collection: str, reason: str)` — NOT a positional string. This was changed during the sub-plan 1 review. See `contracts/errors.py:1060`.

- [ ] **Step 4: Run all RAG transform tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/transforms/rag/ -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/rag/transform.py tests/unit/plugins/transforms/rag/test_transform.py
git commit -m "feat: add readiness guard to RAG transform — refuse empty collection"
```

---

### Task 5: Update All Existing Test Doubles

**Files:**
- Search for all mocks/stubs of `RetrievalProvider` across the test suite

- [ ] **Step 1: Find all existing test doubles**

Run: `grep -r "RetrievalProvider\|spec=.*Provider\|Mock.*provider" tests/ --include="*.py" -l`

For each file found, verify the mock/stub includes `check_readiness`. If using `Mock(spec=RetrievalProvider)`, it auto-generates the method but returns a `Mock`, not a `CollectionReadinessResult`. Switch to `Mock(spec_set=RetrievalProvider)` and configure `check_readiness.return_value` explicitly.

- [ ] **Step 2: Update test doubles**

For each mock that needs updating, add:

```python
mock_provider.check_readiness.return_value = CollectionReadinessResult(
    collection="test",
    reachable=True,
    count=10,
    message="Collection 'test' has 10 documents",
)
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: PASS (no regressions)

- [ ] **Step 4: Commit**

```bash
git add <list specific test files found in Step 1>
git commit -m "fix: update all RetrievalProvider test doubles with check_readiness()"
```

---

### Task 6: Type Checking, Linting, Final Verification

**Files:** None new — verification only.

- [ ] **Step 1: Run type checker on modified files**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/infrastructure/clients/retrieval/base.py src/elspeth/plugins/infrastructure/clients/retrieval/chroma.py src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py src/elspeth/plugins/transforms/rag/transform.py`
Expected: PASS

- [ ] **Step 2: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/infrastructure/clients/retrieval/ src/elspeth/plugins/transforms/rag/`
Expected: PASS

- [ ] **Step 3: Run tier model enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS

- [ ] **Step 4: Run full integration test suite**

Run: `.venv/bin/python -m pytest tests/integration/ -x -q`
Expected: PASS

- [ ] **Step 5: Commit if any fixes needed**

```bash
git add <list specific files with issues>
git commit -m "fix: address type/lint/tier issues in readiness contract"
```
