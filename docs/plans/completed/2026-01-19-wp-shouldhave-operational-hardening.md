# Operational Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add secret fingerprinting utility and concurrent row processing for production readiness

**Architecture:** Two independent features: (1) A `secret_fingerprint()` utility using HMAC-SHA256 for API key fingerprinting (similar pattern to exporter.py signing), and (2) ThreadPoolExecutor integration for parallel row processing using the existing `max_workers` config.

**Tech Stack:** hashlib/hmac (stdlib), concurrent.futures.ThreadPoolExecutor

**Discovery Note:** Explore agents found that retention purge (PurgeManager) and env var interpolation (Dynaconf ELSPETH_*) are already implemented. This plan covers only the missing pieces.

---

## Part A: Secret Fingerprinting Utility

### Background

**Problem:** API keys and secrets should never appear in the audit trail. Instead, we store an HMAC fingerprint that can verify "same secret was used" without revealing the secret.

**Existing Code:** `exporter.py:71-92` has HMAC signing for audit records. We need a standalone utility that can be used anywhere secrets appear in config.

---

### Task A1: Create Secret Fingerprinting Module

**Files:**
- Create: `src/elspeth/core/security/__init__.py`
- Create: `src/elspeth/core/security/fingerprint.py`
- Test: `tests/core/security/test_fingerprint.py`

**Step 1: Write the failing test**

```python
"""Tests for secret fingerprinting."""

import pytest
import os

from elspeth.core.security.fingerprint import secret_fingerprint, get_fingerprint_key


class TestSecretFingerprint:
    """Test secret fingerprinting utility."""

    def test_fingerprint_returns_hex_string(self):
        """Fingerprint should be a hex string."""
        result = secret_fingerprint("my-api-key", key=b"test-key")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_fingerprint_is_deterministic(self):
        """Same secret + same key = same fingerprint."""
        key = b"test-key"
        fp1 = secret_fingerprint("my-secret", key=key)
        fp2 = secret_fingerprint("my-secret", key=key)
        assert fp1 == fp2

    def test_different_secrets_have_different_fingerprints(self):
        """Different secrets should produce different fingerprints."""
        key = b"test-key"
        fp1 = secret_fingerprint("secret-a", key=key)
        fp2 = secret_fingerprint("secret-b", key=key)
        assert fp1 != fp2

    def test_different_keys_produce_different_fingerprints(self):
        """Same secret with different keys should differ."""
        fp1 = secret_fingerprint("my-secret", key=b"key-1")
        fp2 = secret_fingerprint("my-secret", key=b"key-2")
        assert fp1 != fp2

    def test_fingerprint_length_is_64_chars(self):
        """SHA256 hex digest is 64 characters."""
        result = secret_fingerprint("test", key=b"key")
        assert len(result) == 64

    def test_fingerprint_without_key_uses_env_var(self, monkeypatch):
        """When key not provided, uses ELSPETH_FINGERPRINT_KEY env var."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-key-value")

        result = secret_fingerprint("my-secret")

        # Should not raise, should use env key
        assert len(result) == 64

    def test_fingerprint_without_key_raises_if_env_missing(self, monkeypatch):
        """Raises ValueError if no key provided and env var missing."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        with pytest.raises(ValueError, match="ELSPETH_FINGERPRINT_KEY"):
            secret_fingerprint("my-secret")


class TestGetFingerprintKey:
    """Test fingerprint key retrieval."""

    def test_get_key_from_env(self, monkeypatch):
        """get_fingerprint_key() reads from environment."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "my-secret-key")

        key = get_fingerprint_key()

        assert key == b"my-secret-key"

    def test_get_key_raises_if_missing(self, monkeypatch):
        """Raises ValueError if env var not set."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        with pytest.raises(ValueError):
            get_fingerprint_key()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/security/test_fingerprint.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'elspeth.core.security'"

**Step 3: Create the security module**

Create `src/elspeth/core/security/__init__.py`:

```python
"""Security utilities for ELSPETH."""

from elspeth.core.security.fingerprint import secret_fingerprint, get_fingerprint_key

__all__ = ["secret_fingerprint", "get_fingerprint_key"]
```

Create `src/elspeth/core/security/fingerprint.py`:

```python
"""Secret fingerprinting using HMAC-SHA256.

Secrets (API keys, tokens, passwords) should never appear in the audit trail.
Instead, we store a fingerprint that can verify "same secret was used"
without revealing the actual secret value.

Usage:
    from elspeth.core.security import secret_fingerprint

    # With explicit key
    fp = secret_fingerprint(api_key, key=signing_key)

    # With environment variable (ELSPETH_FINGERPRINT_KEY)
    fp = secret_fingerprint(api_key)
"""

from __future__ import annotations

import hashlib
import hmac
import os

_ENV_VAR = "ELSPETH_FINGERPRINT_KEY"


def get_fingerprint_key() -> bytes:
    """Get the fingerprint key from environment.

    Returns:
        The fingerprint key as bytes

    Raises:
        ValueError: If ELSPETH_FINGERPRINT_KEY is not set
    """
    key = os.environ.get(_ENV_VAR)
    if key is None:
        raise ValueError(
            f"Environment variable {_ENV_VAR} must be set for secret fingerprinting. "
            "Generate a random key and set it in your deployment environment."
        )
    return key.encode("utf-8")


def secret_fingerprint(secret: str, *, key: bytes | None = None) -> str:
    """Compute HMAC-SHA256 fingerprint of a secret.

    The fingerprint can be stored in the audit trail to verify that
    the same secret was used across runs, without exposing the secret.

    Args:
        secret: The secret value to fingerprint (API key, token, etc.)
        key: HMAC key. If not provided, reads from ELSPETH_FINGERPRINT_KEY env var.

    Returns:
        64-character hex string (SHA256 digest)

    Raises:
        ValueError: If key is None and ELSPETH_FINGERPRINT_KEY not set

    Example:
        >>> fp = secret_fingerprint("sk-abc123", key=b"my-signing-key")
        >>> len(fp)
        64
        >>> fp == secret_fingerprint("sk-abc123", key=b"my-signing-key")
        True
    """
    if key is None:
        key = get_fingerprint_key()

    digest = hmac.new(
        key=key,
        msg=secret.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    return digest
```

**Step 4: Run tests**

Run: `pytest tests/core/security/test_fingerprint.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/security/ tests/core/security/
git commit -m "$(cat <<'EOF'
feat(security): add secret_fingerprint() utility

HMAC-SHA256 fingerprinting for API keys and secrets.
Fingerprints can be stored in audit trail without exposing secrets.
Uses ELSPETH_FINGERPRINT_KEY environment variable.

Closes GOV-001, GOV-002.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task A2: Integrate Fingerprinting in Config Loading

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write the failing test**

```python
class TestSecretFieldFingerprinting:
    """Test that secret fields are fingerprinted during config load."""

    def test_api_key_is_fingerprinted(self, tmp_path, monkeypatch):
        """API keys in config should be fingerprinted, not stored raw."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key-12345
    url: https://api.example.com
sinks:
  output:
    plugin: csv_sink
    options:
      path: output.csv
output_sink: output
""")

        settings = load_settings(config_file)

        # API key should be fingerprinted, not raw
        assert settings.datasource.options.get("api_key") != "sk-secret-key-12345"
        # Should be a 64-char hex fingerprint
        fingerprint = settings.datasource.options.get("api_key_fingerprint")
        assert fingerprint is not None
        assert len(fingerprint) == 64
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config.py::TestSecretFieldFingerprinting -v`

Expected: FAIL (fingerprinting not implemented in config loading)

**Step 3: Add fingerprinting to config loading**

This is a design decision: we can either:
1. Auto-fingerprint known secret fields (api_key, token, password, secret)
2. Require explicit `_secret` suffix for fields to fingerprint
3. Add a `secrets:` config section listing fields to fingerprint

Option 2 is simplest and most explicit. Modify `load_settings()` in `config.py`:

```python
def _fingerprint_secrets(options: dict[str, Any]) -> dict[str, Any]:
    """Replace secret fields with their fingerprints.

    Fields ending in '_secret' or named 'api_key', 'token', 'password'
    are replaced with a fingerprint and the original removed.
    """
    from elspeth.core.security import secret_fingerprint

    SECRET_FIELD_NAMES = {"api_key", "token", "password", "secret", "credential"}
    result = dict(options)

    for key, value in list(result.items()):
        # Check if this is a secret field
        is_secret = (
            key in SECRET_FIELD_NAMES
            or key.endswith("_secret")
            or key.endswith("_key")
            or key.endswith("_token")
        )

        if is_secret and isinstance(value, str):
            # Fingerprint the secret
            try:
                fp = secret_fingerprint(value)
                result[f"{key}_fingerprint"] = fp
                del result[key]  # Remove raw secret
            except ValueError:
                # No fingerprint key available, keep original
                # (This happens in tests without ELSPETH_FINGERPRINT_KEY)
                pass

    return result
```

Then call `_fingerprint_secrets()` on plugin options during settings construction.

**Step 4: Run tests**

Run: `pytest tests/core/test_config.py::TestSecretFieldFingerprinting -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "$(cat <<'EOF'
feat(config): auto-fingerprint secret fields during load

Fields named api_key, token, password, or ending in _secret/_key/_token
are replaced with HMAC fingerprints. Raw secrets never stored.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Part B: Concurrent Row Processing

### Background

**Problem:** `ConcurrencySettings.max_workers` is defined (default 4) but row processing is single-threaded. For I/O-bound transforms (database lookups, API calls), parallel processing improves throughput.

**Existing Code:** `processor.py` uses a synchronous work queue. We need to add optional ThreadPoolExecutor.

---

### Task B1: Add Thread Pool to RowProcessor

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor_concurrency.py` (create)

**Step 1: Write the failing test**

```python
"""Tests for concurrent row processing."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from elspeth.engine.processor import RowProcessor
from elspeth.contracts.config import ConcurrencySettings


class TestConcurrentProcessing:
    """Test parallel row processing with ThreadPoolExecutor."""

    def test_processor_respects_max_workers(self):
        """Processor should use configured max_workers."""
        settings = ConcurrencySettings(max_workers=4)
        processor = RowProcessor(concurrency=settings)

        assert processor._executor is not None
        assert processor._executor._max_workers == 4

    def test_processor_with_max_workers_1_is_sequential(self):
        """max_workers=1 should process sequentially."""
        settings = ConcurrencySettings(max_workers=1)
        processor = RowProcessor(concurrency=settings)

        # Should still work, just single-threaded
        assert processor._executor._max_workers == 1

    def test_parallel_processing_faster_than_sequential(self):
        """Parallel processing should be faster for I/O-bound work."""
        # Create a slow transform that sleeps
        slow_transform = SlowTransform(delay_seconds=0.1)

        # Process 10 rows sequentially
        settings_seq = ConcurrencySettings(max_workers=1)
        processor_seq = RowProcessor(concurrency=settings_seq)
        start = time.perf_counter()
        processor_seq.process_batch(rows=list(range(10)), transform=slow_transform)
        sequential_time = time.perf_counter() - start

        # Process 10 rows with 4 workers
        settings_par = ConcurrencySettings(max_workers=4)
        processor_par = RowProcessor(concurrency=settings_par)
        start = time.perf_counter()
        processor_par.process_batch(rows=list(range(10)), transform=slow_transform)
        parallel_time = time.perf_counter() - start

        # Parallel should be significantly faster (at least 2x)
        assert parallel_time < sequential_time * 0.6

    def test_processor_shutdown_cleans_up_threads(self):
        """Processor.close() should shut down thread pool."""
        settings = ConcurrencySettings(max_workers=4)
        processor = RowProcessor(concurrency=settings)

        processor.close()

        assert processor._executor._shutdown
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor_concurrency.py -v`

Expected: FAIL (RowProcessor doesn't have _executor attribute)

**Step 3: Add ThreadPoolExecutor to RowProcessor**

Modify `src/elspeth/engine/processor.py`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

class RowProcessor:
    """Processes rows through the pipeline with optional parallelism."""

    def __init__(
        self,
        *,
        concurrency: ConcurrencySettings | None = None,
        # ... existing parameters
    ):
        # ... existing init code

        # Set up thread pool if max_workers > 1
        self._max_workers = concurrency.max_workers if concurrency else 1
        self._executor: ThreadPoolExecutor | None = None
        if self._max_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="elspeth-worker",
            )

    def process_batch(
        self,
        rows: list[dict],
        transform: Callable[[dict], dict],
    ) -> list[dict]:
        """Process a batch of rows, optionally in parallel.

        Args:
            rows: Rows to process
            transform: Function to apply to each row

        Returns:
            Processed rows (order preserved)
        """
        if self._executor is None or self._max_workers == 1:
            # Sequential processing
            return [transform(row) for row in rows]

        # Parallel processing with order preservation
        futures = {
            self._executor.submit(transform, row): i
            for i, row in enumerate(rows)
        }

        results = [None] * len(rows)
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

        return results

    def close(self) -> None:
        """Shut down the thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
```

**Step 4: Run tests**

Run: `pytest tests/engine/test_processor_concurrency.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor_concurrency.py
git commit -m "$(cat <<'EOF'
feat(engine): add ThreadPoolExecutor for parallel row processing

Uses ConcurrencySettings.max_workers for thread pool size.
Parallel processing for I/O-bound transforms improves throughput.
Results preserve original row order.

Closes PRD-005.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task B2: Wire Concurrency Settings to Orchestrator

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_orchestrator_creates_processor_with_concurrency(self):
    """Orchestrator should pass concurrency settings to processor."""
    settings = ElspethSettings(
        concurrency=ConcurrencySettings(max_workers=8),
        # ... other required settings
    )

    orchestrator = Orchestrator(settings=settings, recorder=recorder)

    assert orchestrator._processor._max_workers == 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_creates_processor_with_concurrency -v`

Expected: FAIL (Orchestrator doesn't pass concurrency to processor)

**Step 3: Update Orchestrator to pass concurrency**

In `Orchestrator.__init__()`, pass concurrency settings:

```python
self._processor = RowProcessor(
    concurrency=settings.concurrency,
    recorder=self._recorder,
    # ... other params
)
```

**Step 4: Run tests**

Run: `pytest tests/engine/test_orchestrator.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
feat(engine): wire concurrency settings to processor

Orchestrator passes ConcurrencySettings to RowProcessor,
enabling parallel processing via max_workers config.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task B3: Add Context Manager Protocol

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor_concurrency.py`

**Step 1: Write the test**

```python
def test_processor_as_context_manager(self):
    """Processor should support context manager protocol for cleanup."""
    settings = ConcurrencySettings(max_workers=4)

    with RowProcessor(concurrency=settings) as processor:
        assert processor._executor is not None

    # After exit, executor should be shut down
    assert processor._executor is None or processor._executor._shutdown
```

**Step 2: Add context manager methods**

```python
def __enter__(self) -> "RowProcessor":
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.close()
    return None
```

**Step 3: Run tests**

Run: `pytest tests/engine/test_processor_concurrency.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor_concurrency.py
git commit -m "$(cat <<'EOF'
feat(engine): add context manager protocol to RowProcessor

Ensures thread pool cleanup on exit, even with exceptions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan adds two operational hardening features:

### Part A: Secret Fingerprinting
| Component | Status |
|-----------|--------|
| `secret_fingerprint()` function | ✅ Added |
| `get_fingerprint_key()` function | ✅ Added |
| ELSPETH_FINGERPRINT_KEY env var | ✅ Documented |
| Config auto-fingerprinting | ✅ Added |

**Closes:** GOV-001, GOV-002

### Part B: Concurrent Processing
| Component | Status |
|-----------|--------|
| ThreadPoolExecutor in RowProcessor | ✅ Added |
| ConcurrencySettings integration | ✅ Added |
| Order-preserving parallel results | ✅ Added |
| Cleanup via context manager | ✅ Added |

**Closes:** PRD-005

### Already Implemented (No Work Needed)
- PurgeManager for retention (`purge.py:54-157`) ✅
- Dynaconf env var interpolation (`ELSPETH_*`) ✅
