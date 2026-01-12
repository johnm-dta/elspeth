# Phase 6: External Calls (Tasks 1-14)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add infrastructure for recording, replaying, and verifying external calls (LLM, HTTP, etc.). This enables reproducibility and audit compliance for non-deterministic operations.

**Architecture:** External calls are recorded in the `calls` table with request/response hashes and PayloadStore refs. Run modes (live/replay/verify) control whether calls execute live, use recorded responses, or compare both.

**Tech Stack:** Python 3.11+, LiteLLM (LLM providers), DeepDiff (verification), httpx (HTTP calls)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.payload_store`
- Phase 3A: `elspeth.core.landscape` (schema includes `calls` table)
- Phase 5: `elspeth.core.rate_limit` (for rate limiting external calls)

---

## Auditability Requirement

**Every external call must be fully recorded:**

1. **Request** - Full request body/params stored in PayloadStore, hash in `calls.request_hash`
2. **Response** - Full response stored in PayloadStore, hash in `calls.response_hash`
3. **Metadata** - Provider, latency, status, errors in `calls` table
4. **Secrets** - NEVER stored; only HMAC fingerprints for "same key used" verification

The `calls` table schema (from Phase 3A):
```sql
CREATE TABLE calls (
    call_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL REFERENCES node_states(state_id),
    call_index INTEGER NOT NULL,
    call_type TEXT NOT NULL,               -- llm, http, sql, filesystem
    status TEXT NOT NULL,                  -- success, error
    request_hash TEXT NOT NULL,
    request_ref TEXT,
    response_hash TEXT,
    response_ref TEXT,
    error_json TEXT,
    latency_ms REAL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE(state_id, call_index)
);
```

---

## Task 1: CallRecorder - Record External Calls

**Context:** Create CallRecorder service that records external call request/response pairs to Landscape and PayloadStore.

**Files:**
- Create: `src/elspeth/core/calls/__init__.py`
- Create: `src/elspeth/core/calls/recorder.py`
- Create: `tests/core/calls/__init__.py`
- Create: `tests/core/calls/test_recorder.py`

### Step 1: Write the failing test

```python
# tests/core/calls/__init__.py
"""External call tests."""

# tests/core/calls/test_recorder.py
"""Tests for CallRecorder."""

import pytest
from datetime import datetime, timezone


class TestCallRecorder:
    """Tests for external call recording."""

    @pytest.fixture
    def recorder(self, landscape_db, payload_store):
        """Create CallRecorder for tests."""
        from elspeth.core.calls import CallRecorder
        return CallRecorder(landscape_db, payload_store)

    def test_record_successful_call(self, recorder, node_state_id) -> None:
        """Can record a successful external call."""
        call = recorder.record_call(
            state_id=node_state_id,
            call_index=0,
            call_type="llm",
            request={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            response={"choices": [{"message": {"content": "Hi there!"}}]},
            latency_ms=150.5,
        )

        assert call.call_id is not None
        assert call.status == "success"
        assert call.request_hash is not None
        assert call.response_hash is not None

    def test_record_failed_call(self, recorder, node_state_id) -> None:
        """Can record a failed external call."""
        call = recorder.record_call(
            state_id=node_state_id,
            call_index=0,
            call_type="http",
            request={"url": "https://api.example.com/data"},
            response=None,
            error={"code": 500, "message": "Internal Server Error"},
            latency_ms=50.0,
        )

        assert call.status == "error"
        assert call.response_hash is None
        assert call.error_json is not None

    def test_request_stored_in_payload_store(self, recorder, payload_store, node_state_id) -> None:
        """Request body is stored in PayloadStore."""
        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}

        call = recorder.record_call(
            state_id=node_state_id,
            call_index=0,
            call_type="llm",
            request=request,
            response={"choices": []},
            latency_ms=100.0,
        )

        # Verify payload is retrievable
        stored = payload_store.retrieve(call.request_ref)
        assert stored is not None

    def test_get_calls_for_state(self, recorder, node_state_id) -> None:
        """Can retrieve all calls for a node_state."""
        # Record multiple calls
        recorder.record_call(node_state_id, 0, "llm", {"req": 1}, {"resp": 1}, 100)
        recorder.record_call(node_state_id, 1, "http", {"req": 2}, {"resp": 2}, 50)

        calls = recorder.get_calls_for_state(node_state_id)

        assert len(calls) == 2
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1
```

### Step 2: Implementation

```python
# src/elspeth/core/calls/__init__.py
"""External call recording and replay.

Provides:
- CallRecorder: Record request/response pairs
- CallReplayer: Replay recorded responses
- CallVerifier: Compare live vs recorded
"""

from elspeth.core.calls.recorder import CallRecorder

__all__ = ["CallRecorder"]


# src/elspeth/core/calls/recorder.py
"""CallRecorder for recording external calls."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from elspeth.core.canonical import canonical_json


@dataclass
class RecordedCall:
    """A recorded external call."""
    call_id: str
    state_id: str
    call_index: int
    call_type: str
    status: str
    request_hash: str
    request_ref: str | None
    response_hash: str | None
    response_ref: str | None
    error_json: str | None
    latency_ms: float | None
    created_at: datetime


class CallRecorder:
    """Records external calls to Landscape and PayloadStore."""

    def __init__(self, db, payload_store) -> None:
        self._db = db
        self._payload_store = payload_store

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: str,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        latency_ms: float | None = None,
        error: dict[str, Any] | None = None,
    ) -> RecordedCall:
        """Record an external call."""
        from elspeth.core.landscape.schema import calls_table

        call_id = f"call-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Compute hashes and store payloads
        request_canonical = canonical_json(request)
        request_hash = self._payload_store.compute_hash(request_canonical.encode())
        request_ref = self._payload_store.store(request_canonical.encode())

        response_hash = None
        response_ref = None
        if response is not None:
            response_canonical = canonical_json(response)
            response_hash = self._payload_store.compute_hash(response_canonical.encode())
            response_ref = self._payload_store.store(response_canonical.encode())

        error_json = json.dumps(error) if error else None
        status = "error" if error else "success"

        with self._db.engine.connect() as conn:
            conn.execute(
                calls_table.insert().values(
                    call_id=call_id,
                    state_id=state_id,
                    call_index=call_index,
                    call_type=call_type,
                    status=status,
                    request_hash=request_hash,
                    request_ref=request_ref,
                    response_hash=response_hash,
                    response_ref=response_ref,
                    error_json=error_json,
                    latency_ms=latency_ms,
                    created_at=now,
                )
            )
            conn.commit()

        return RecordedCall(
            call_id=call_id,
            state_id=state_id,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash=request_hash,
            request_ref=request_ref,
            response_hash=response_hash,
            response_ref=response_ref,
            error_json=error_json,
            latency_ms=latency_ms,
            created_at=now,
        )

    def get_calls_for_state(self, state_id: str) -> list[RecordedCall]:
        """Get all calls for a node_state."""
        from sqlalchemy import select, asc
        from elspeth.core.landscape.schema import calls_table

        with self._db.engine.connect() as conn:
            results = conn.execute(
                select(calls_table)
                .where(calls_table.c.state_id == state_id)
                .order_by(asc(calls_table.c.call_index))
            ).fetchall()

        return [
            RecordedCall(
                call_id=r.call_id,
                state_id=r.state_id,
                call_index=r.call_index,
                call_type=r.call_type,
                status=r.status,
                request_hash=r.request_hash,
                request_ref=r.request_ref,
                response_hash=r.response_hash,
                response_ref=r.response_ref,
                error_json=r.error_json,
                latency_ms=r.latency_ms,
                created_at=r.created_at,
            )
            for r in results
        ]
```

### Step 3: Run tests

Run: `pytest tests/core/calls/test_recorder.py -v`
Expected: PASS

---

## Task 2: Run Mode Configuration

**Context:** Add run mode (live/replay/verify) to configuration and RunContext.

**Files:**
- Modify: `src/elspeth/core/config.py` (add RunMode, ExternalCallSettings)
- Modify: `tests/core/test_config.py` (add run mode tests)

### Step 1: Write the failing test

```python
# Add to tests/core/test_config.py

class TestRunModeSettings:
    """Tests for run mode configuration."""

    def test_run_mode_defaults_to_live(self) -> None:
        from elspeth.core.config import ExternalCallSettings, RunMode

        settings = ExternalCallSettings()

        assert settings.mode == RunMode.LIVE

    def test_run_mode_options(self) -> None:
        from elspeth.core.config import ExternalCallSettings, RunMode

        live = ExternalCallSettings(mode=RunMode.LIVE)
        replay = ExternalCallSettings(mode=RunMode.REPLAY)
        verify = ExternalCallSettings(mode=RunMode.VERIFY)

        assert live.mode == RunMode.LIVE
        assert replay.mode == RunMode.REPLAY
        assert verify.mode == RunMode.VERIFY

    def test_replay_mode_requires_source_run(self) -> None:
        from pydantic import ValidationError
        from elspeth.core.config import ExternalCallSettings, RunMode

        # Replay needs a source run to replay from
        with pytest.raises(ValidationError):
            ExternalCallSettings(mode=RunMode.REPLAY, replay_source_run_id=None)

    def test_verify_mode_requires_source_run(self) -> None:
        from pydantic import ValidationError
        from elspeth.core.config import ExternalCallSettings, RunMode

        with pytest.raises(ValidationError):
            ExternalCallSettings(mode=RunMode.VERIFY, replay_source_run_id=None)
```

### Step 2: Add RunMode and ExternalCallSettings

Add to `src/elspeth/core/config.py`:

```python
from enum import Enum


class RunMode(Enum):
    """Run mode for external calls."""
    LIVE = "live"        # Execute live, record request/response
    REPLAY = "replay"    # Use recorded responses, no live calls
    VERIFY = "verify"    # Execute live AND compare to recorded


class ExternalCallSettings(BaseSettings):
    """Configuration for external call handling."""

    mode: RunMode = RunMode.LIVE
    replay_source_run_id: str | None = None  # Required for replay/verify
    verify_fail_on_drift: bool = True  # Fail run if verify detects drift
    record_payloads: bool = True  # Store full request/response

    @model_validator(mode="after")
    def validate_replay_source(self) -> "ExternalCallSettings":
        if self.mode in (RunMode.REPLAY, RunMode.VERIFY):
            if self.replay_source_run_id is None:
                raise ValueError(
                    f"replay_source_run_id required when mode={self.mode.value}"
                )
        return self
```

### Step 3: Run tests

Run: `pytest tests/core/test_config.py::TestRunModeSettings -v`
Expected: PASS

---

## Task 3: Secret Fingerprinting

**Context:** Implement HMAC-based secret fingerprinting so we can verify "same secret used" without storing secrets.

**Files:**
- Create: `src/elspeth/core/secrets/__init__.py`
- Create: `src/elspeth/core/secrets/fingerprint.py`
- Create: `tests/core/secrets/__init__.py`
- Create: `tests/core/secrets/test_fingerprint.py`

### Step 1: Write the failing test

```python
# tests/core/secrets/__init__.py
"""Secret handling tests."""

# tests/core/secrets/test_fingerprint.py
"""Tests for secret fingerprinting."""

import pytest


class TestSecretFingerprint:
    """Tests for HMAC-based secret fingerprinting."""

    def test_same_secret_same_fingerprint(self) -> None:
        """Same secret produces same fingerprint."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-fingerprint-key")

        fp1 = fingerprinter.fingerprint("my-api-key-123")
        fp2 = fingerprinter.fingerprint("my-api-key-123")

        assert fp1 == fp2

    def test_different_secrets_different_fingerprints(self) -> None:
        """Different secrets produce different fingerprints."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-fingerprint-key")

        fp1 = fingerprinter.fingerprint("key-A")
        fp2 = fingerprinter.fingerprint("key-B")

        assert fp1 != fp2

    def test_different_keys_different_fingerprints(self) -> None:
        """Different fingerprint keys produce different results."""
        from elspeth.core.secrets import SecretFingerprinter

        fp1 = SecretFingerprinter(key=b"key-1").fingerprint("secret")
        fp2 = SecretFingerprinter(key=b"key-2").fingerprint("secret")

        assert fp1 != fp2

    def test_fingerprint_is_hex_string(self) -> None:
        """Fingerprint is a hex-encoded string."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-key")
        fp = fingerprinter.fingerprint("secret")

        # Should be 64 hex chars (sha256)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_redact_config_replaces_secrets(self) -> None:
        """redact_config replaces secret values with fingerprints."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-key")

        config = {
            "model": "gpt-4",
            "api_key": "sk-secret-key-12345",
            "temperature": 0.7,
        }

        redacted = fingerprinter.redact_config(
            config,
            secret_fields=["api_key"],
        )

        assert redacted["model"] == "gpt-4"
        assert redacted["temperature"] == 0.7
        assert redacted["api_key"] == "[REDACTED]"
        assert "api_key_fingerprint" in redacted
        assert len(redacted["api_key_fingerprint"]) == 64
```

### Step 2: Create SecretFingerprinter

```python
# src/elspeth/core/secrets/__init__.py
"""Secret handling utilities.

NEVER stores actual secrets. Only HMAC fingerprints for verification.
"""

from elspeth.core.secrets.fingerprint import SecretFingerprinter

__all__ = ["SecretFingerprinter"]


# src/elspeth/core/secrets/fingerprint.py
"""HMAC-based secret fingerprinting."""

import hmac
import hashlib
from typing import Any


class SecretFingerprinter:
    """Generates HMAC fingerprints for secrets.

    Uses HMAC (not plain hash) to prevent offline guessing attacks.
    An attacker would need both the fingerprint AND the key to verify.

    Example:
        fingerprinter = SecretFingerprinter(key=os.environ["FINGERPRINT_KEY"])
        fp = fingerprinter.fingerprint(api_key)

        # Store fp in Landscape, NEVER store api_key
    """

    def __init__(self, key: bytes) -> None:
        """Initialize with fingerprint key.

        Args:
            key: HMAC key (load from env/secrets manager, never from Landscape)
        """
        self._key = key

    def fingerprint(self, secret: str) -> str:
        """Generate fingerprint for a secret value.

        Args:
            secret: The secret value to fingerprint

        Returns:
            Hex-encoded HMAC-SHA256 fingerprint
        """
        return hmac.new(
            self._key,
            secret.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def redact_config(
        self,
        config: dict[str, Any],
        secret_fields: list[str],
    ) -> dict[str, Any]:
        """Redact secret fields and add fingerprints.

        Args:
            config: Configuration dict
            secret_fields: Field names containing secrets

        Returns:
            New dict with secrets replaced by [REDACTED] and fingerprints added
        """
        result = dict(config)

        for field in secret_fields:
            if field in result and result[field]:
                # Add fingerprint
                result[f"{field}_fingerprint"] = self.fingerprint(str(result[field]))
                # Redact original
                result[field] = "[REDACTED]"

        return result

    def verify_fingerprint(self, secret: str, fingerprint: str) -> bool:
        """Verify a secret matches a fingerprint.

        Args:
            secret: Secret value to check
            fingerprint: Expected fingerprint

        Returns:
            True if fingerprint matches
        """
        return hmac.compare_digest(
            self.fingerprint(secret),
            fingerprint,
        )
```

### Step 3: Run tests

Run: `pytest tests/core/secrets/test_fingerprint.py -v`
Expected: PASS

---

## Task 4: CallReplayer - Replay Recorded Responses

**Context:** Create CallReplayer that returns recorded responses instead of making live calls.

**Files:**
- Create: `src/elspeth/core/calls/replayer.py`
- Create: `tests/core/calls/test_replayer.py`

### Step 1: Write the failing test

```python
# tests/core/calls/test_replayer.py
"""Tests for CallReplayer."""

import pytest


class TestCallReplayer:
    """Tests for replaying recorded calls."""

    @pytest.fixture
    def replayer(self, landscape_db, payload_store):
        """Create CallReplayer for tests."""
        from elspeth.core.calls import CallReplayer
        return CallReplayer(landscape_db, payload_store)

    def test_get_recorded_response(
        self, replayer, recorded_call_fixture
    ) -> None:
        """Can retrieve recorded response for matching request."""
        response = replayer.get_recorded_response(
            source_run_id=recorded_call_fixture.run_id,
            request_hash=recorded_call_fixture.request_hash,
        )

        assert response is not None
        assert "choices" in response  # LLM response structure

    def test_replay_returns_none_for_unknown_request(self, replayer) -> None:
        """Returns None when no matching recorded call exists."""
        response = replayer.get_recorded_response(
            source_run_id="nonexistent-run",
            request_hash="unknown-hash",
        )

        assert response is None

    def test_replay_by_call_index(
        self, replayer, state_with_multiple_calls
    ) -> None:
        """Can replay by state_id and call_index."""
        response = replayer.get_response_by_index(
            source_state_id=state_with_multiple_calls,
            call_index=0,
        )

        assert response is not None
```

### Step 2: Create CallReplayer

```python
# src/elspeth/core/calls/replayer.py
"""CallReplayer for replay mode."""

import json
from typing import Any

from sqlalchemy import select, and_


class CallReplayer:
    """Retrieves recorded responses for replay mode."""

    def __init__(self, db, payload_store) -> None:
        self._db = db
        self._payload_store = payload_store

    def get_recorded_response(
        self,
        source_run_id: str,
        request_hash: str,
    ) -> dict[str, Any] | None:
        """Get recorded response matching request hash.

        Args:
            source_run_id: Run to replay from
            request_hash: Hash of request to match

        Returns:
            Recorded response dict, or None if not found
        """
        from elspeth.core.landscape.schema import (
            calls_table, node_states_table, tokens_table,
            rows_table,
        )

        with self._db.engine.connect() as conn:
            # Find call with matching request hash in source run
            result = conn.execute(
                select(calls_table.c.response_ref)
                .select_from(
                    calls_table
                    .join(node_states_table,
                          calls_table.c.state_id == node_states_table.c.state_id)
                    .join(tokens_table,
                          node_states_table.c.token_id == tokens_table.c.token_id)
                    .join(rows_table,
                          tokens_table.c.row_id == rows_table.c.row_id)
                )
                .where(and_(
                    rows_table.c.run_id == source_run_id,
                    calls_table.c.request_hash == request_hash,
                    calls_table.c.status == "success",
                ))
                .limit(1)
            ).fetchone()

        if result is None or result.response_ref is None:
            return None

        # Load response from PayloadStore
        payload = self._payload_store.retrieve(result.response_ref)
        if payload is None:
            return None

        return json.loads(payload)

    def get_response_by_index(
        self,
        source_state_id: str,
        call_index: int,
    ) -> dict[str, Any] | None:
        """Get recorded response by state and index.

        Args:
            source_state_id: Source node_state_id
            call_index: Index of call within state

        Returns:
            Recorded response dict, or None if not found
        """
        from elspeth.core.landscape.schema import calls_table

        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(calls_table.c.response_ref)
                .where(and_(
                    calls_table.c.state_id == source_state_id,
                    calls_table.c.call_index == call_index,
                    calls_table.c.status == "success",
                ))
            ).fetchone()

        if result is None or result.response_ref is None:
            return None

        payload = self._payload_store.retrieve(result.response_ref)
        if payload is None:
            return None

        return json.loads(payload)
```

Update `__init__.py`:

```python
from elspeth.core.calls.recorder import CallRecorder
from elspeth.core.calls.replayer import CallReplayer

__all__ = ["CallRecorder", "CallReplayer"]
```

### Step 3: Run tests

Run: `pytest tests/core/calls/test_replayer.py -v`
Expected: PASS

---

## Task 5: CallVerifier - Verify Mode with DeepDiff

**Context:** Create CallVerifier that executes live calls and compares against recorded responses using DeepDiff.

**Files:**
- Create: `src/elspeth/core/calls/verifier.py`
- Create: `tests/core/calls/test_verifier.py`

### Step 1: Write the failing test

```python
# tests/core/calls/test_verifier.py
"""Tests for CallVerifier."""

import pytest


class TestCallVerifier:
    """Tests for verify mode."""

    @pytest.fixture
    def verifier(self, landscape_db, payload_store):
        """Create CallVerifier for tests."""
        from elspeth.core.calls import CallVerifier
        return CallVerifier(landscape_db, payload_store)

    def test_verify_identical_responses(self, verifier) -> None:
        """Identical responses pass verification."""
        recorded = {"choices": [{"message": {"content": "Hello"}}]}
        live = {"choices": [{"message": {"content": "Hello"}}]}

        result = verifier.verify(recorded, live)

        assert result.matched is True
        assert result.diff is None

    def test_verify_different_responses(self, verifier) -> None:
        """Different responses fail verification with diff."""
        recorded = {"choices": [{"message": {"content": "Hello"}}]}
        live = {"choices": [{"message": {"content": "Hi there"}}]}

        result = verifier.verify(recorded, live)

        assert result.matched is False
        assert result.diff is not None

    def test_verify_ignores_non_deterministic_fields(self, verifier) -> None:
        """Non-deterministic fields (id, created) are ignored."""
        recorded = {
            "id": "chatcmpl-abc",
            "created": 1234567890,
            "choices": [{"message": {"content": "Hello"}}],
        }
        live = {
            "id": "chatcmpl-xyz",  # Different ID
            "created": 1234567999,  # Different timestamp
            "choices": [{"message": {"content": "Hello"}}],
        }

        result = verifier.verify(
            recorded, live,
            ignore_paths=["root['id']", "root['created']"],
        )

        assert result.matched is True

    def test_record_drift(self, verifier, call_id) -> None:
        """Drift is recorded in Landscape."""
        recorded = {"content": "A"}
        live = {"content": "B"}

        result = verifier.verify(recorded, live)
        verifier.record_drift(call_id, result)

        # Drift should be queryable
        drift = verifier.get_drift_for_call(call_id)
        assert drift is not None
```

### Step 2: Create CallVerifier

```python
# src/elspeth/core/calls/verifier.py
"""CallVerifier for verify mode using DeepDiff."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from deepdiff import DeepDiff


@dataclass
class VerifyResult:
    """Result of comparing recorded vs live response."""
    matched: bool
    diff: dict[str, Any] | None
    severity: str | None  # info, warning, error


class CallVerifier:
    """Compares live responses against recorded using DeepDiff."""

    # Default paths to ignore (non-deterministic)
    DEFAULT_IGNORE_PATHS = [
        "root['id']",
        "root['created']",
        "root['system_fingerprint']",
    ]

    def __init__(self, db, payload_store) -> None:
        self._db = db
        self._payload_store = payload_store

    def verify(
        self,
        recorded: dict[str, Any],
        live: dict[str, Any],
        ignore_paths: list[str] | None = None,
    ) -> VerifyResult:
        """Compare recorded and live responses.

        Args:
            recorded: Previously recorded response
            live: Current live response
            ignore_paths: Paths to exclude from comparison

        Returns:
            VerifyResult with match status and diff details
        """
        exclude = ignore_paths or self.DEFAULT_IGNORE_PATHS

        diff = DeepDiff(
            recorded,
            live,
            ignore_order=True,
            exclude_paths=exclude,
        )

        if not diff:
            return VerifyResult(matched=True, diff=None, severity=None)

        severity = self._classify_drift(diff)

        return VerifyResult(
            matched=False,
            diff=diff.to_dict(),
            severity=severity,
        )

    def _classify_drift(self, diff: DeepDiff) -> str:
        """Classify drift severity."""
        # Type changes or removed keys are errors
        if "type_changes" in diff or "dictionary_item_removed" in diff:
            return "error"

        # Value changes in content are warnings
        if "values_changed" in diff:
            return "warning"

        # Added keys are usually info
        return "info"

    def record_drift(self, call_id: str, result: VerifyResult) -> None:
        """Record verification drift in Landscape.

        Args:
            call_id: The call that was verified
            result: Verification result with diff
        """
        if result.matched:
            return

        from elspeth.core.landscape.schema import calls_table
        from sqlalchemy import update

        # Store drift in error_json field (repurposed for verify mode)
        drift_record = {
            "drift_detected": True,
            "severity": result.severity,
            "diff": result.diff,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._db.engine.connect() as conn:
            conn.execute(
                update(calls_table)
                .where(calls_table.c.call_id == call_id)
                .values(error_json=json.dumps(drift_record))
            )
            conn.commit()

    def get_drift_for_call(self, call_id: str) -> dict[str, Any] | None:
        """Get recorded drift for a call."""
        from elspeth.core.landscape.schema import calls_table
        from sqlalchemy import select

        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(calls_table.c.error_json)
                .where(calls_table.c.call_id == call_id)
            ).fetchone()

        if result is None or result.error_json is None:
            return None

        data = json.loads(result.error_json)
        if data.get("drift_detected"):
            return data

        return None
```

Update `__init__.py`:

```python
from elspeth.core.calls.recorder import CallRecorder
from elspeth.core.calls.replayer import CallReplayer
from elspeth.core.calls.verifier import CallVerifier, VerifyResult

__all__ = ["CallRecorder", "CallReplayer", "CallVerifier", "VerifyResult"]
```

### Step 3: Run tests

Run: `pytest tests/core/calls/test_verifier.py -v`
Expected: PASS

---

## Task 6: ExternalCallWrapper - Mode-Aware Call Execution

**Context:** Create wrapper that executes external calls according to run mode (live/replay/verify).

**Files:**
- Create: `src/elspeth/core/calls/wrapper.py`
- Create: `tests/core/calls/test_wrapper.py`

### Step 1: Write the failing test

```python
# tests/core/calls/test_wrapper.py
"""Tests for ExternalCallWrapper."""

import pytest
from unittest.mock import Mock, AsyncMock


class TestExternalCallWrapper:
    """Tests for mode-aware call execution."""

    def test_live_mode_executes_and_records(self, wrapper_live_mode) -> None:
        """Live mode executes call and records."""
        mock_executor = Mock(return_value={"result": "data"})

        result = wrapper_live_mode.execute(
            executor=mock_executor,
            request={"query": "test"},
            call_type="http",
        )

        mock_executor.assert_called_once()
        assert result == {"result": "data"}

    def test_replay_mode_returns_recorded(self, wrapper_replay_mode) -> None:
        """Replay mode returns recorded response without executing."""
        mock_executor = Mock()

        result = wrapper_replay_mode.execute(
            executor=mock_executor,
            request={"query": "test"},
            call_type="http",
        )

        mock_executor.assert_not_called()
        assert result is not None  # From recorded

    def test_verify_mode_executes_and_compares(self, wrapper_verify_mode) -> None:
        """Verify mode executes and compares to recorded."""
        mock_executor = Mock(return_value={"result": "data"})

        result = wrapper_verify_mode.execute(
            executor=mock_executor,
            request={"query": "test"},
            call_type="http",
        )

        mock_executor.assert_called_once()
        # Result is from live execution
        assert result == {"result": "data"}

    def test_replay_mode_fails_without_recording(
        self, wrapper_replay_mode_no_recording
    ) -> None:
        """Replay mode raises when no recording exists."""
        from elspeth.core.calls.wrapper import ReplayNotFoundError

        mock_executor = Mock()

        with pytest.raises(ReplayNotFoundError):
            wrapper_replay_mode_no_recording.execute(
                executor=mock_executor,
                request={"query": "unknown"},
                call_type="http",
            )
```

### Step 2: Create ExternalCallWrapper

```python
# src/elspeth/core/calls/wrapper.py
"""Mode-aware external call wrapper."""

import time
from typing import Any, Callable

from elspeth.core.config import RunMode
from elspeth.core.calls.recorder import CallRecorder
from elspeth.core.calls.replayer import CallReplayer
from elspeth.core.calls.verifier import CallVerifier
from elspeth.core.canonical import canonical_json


class ReplayNotFoundError(Exception):
    """Raised when replay mode cannot find recorded response."""
    pass


class VerificationDriftError(Exception):
    """Raised when verify mode detects unacceptable drift."""
    pass


class ExternalCallWrapper:
    """Executes external calls according to run mode."""

    def __init__(
        self,
        mode: RunMode,
        recorder: CallRecorder,
        replayer: CallReplayer | None,
        verifier: CallVerifier | None,
        payload_store,
        state_id: str,
        source_run_id: str | None = None,
        fail_on_drift: bool = True,
    ) -> None:
        self._mode = mode
        self._recorder = recorder
        self._replayer = replayer
        self._verifier = verifier
        self._payload_store = payload_store
        self._state_id = state_id
        self._source_run_id = source_run_id
        self._fail_on_drift = fail_on_drift
        self._call_index = 0

    def execute(
        self,
        executor: Callable[[], dict[str, Any]],
        request: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Execute an external call according to mode.

        Args:
            executor: Callable that makes the actual call
            request: Request data (for recording/matching)
            call_type: Type of call (llm, http, etc.)

        Returns:
            Response dict

        Raises:
            ReplayNotFoundError: Replay mode, no recording found
            VerificationDriftError: Verify mode, drift exceeds threshold
        """
        request_hash = self._compute_request_hash(request)

        if self._mode == RunMode.REPLAY:
            return self._execute_replay(request_hash)

        elif self._mode == RunMode.VERIFY:
            return self._execute_verify(executor, request, request_hash, call_type)

        else:  # LIVE
            return self._execute_live(executor, request, call_type)

    def _execute_live(
        self,
        executor: Callable,
        request: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Execute live and record."""
        start = time.monotonic()
        error = None
        response = None

        try:
            response = executor()
        except Exception as e:
            error = {"type": type(e).__name__, "message": str(e)}
            raise
        finally:
            latency = (time.monotonic() - start) * 1000

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=self._call_index,
                call_type=call_type,
                request=request,
                response=response,
                latency_ms=latency,
                error=error,
            )
            self._call_index += 1

        return response

    def _execute_replay(self, request_hash: str) -> dict[str, Any]:
        """Return recorded response without executing."""
        if self._replayer is None or self._source_run_id is None:
            raise ReplayNotFoundError("Replayer not configured")

        response = self._replayer.get_recorded_response(
            source_run_id=self._source_run_id,
            request_hash=request_hash,
        )

        if response is None:
            raise ReplayNotFoundError(
                f"No recorded response for request hash {request_hash}"
            )

        self._call_index += 1
        return response

    def _execute_verify(
        self,
        executor: Callable,
        request: dict[str, Any],
        request_hash: str,
        call_type: str,
    ) -> dict[str, Any]:
        """Execute live and compare to recorded."""
        # Get recorded response
        recorded = None
        if self._replayer and self._source_run_id:
            recorded = self._replayer.get_recorded_response(
                self._source_run_id, request_hash
            )

        # Execute live
        live_response = self._execute_live(executor, request, call_type)

        # Compare if we have recorded
        if recorded is not None and self._verifier is not None:
            result = self._verifier.verify(recorded, live_response)

            if not result.matched:
                # Record drift
                # Note: Would need call_id from record_call
                if self._fail_on_drift and result.severity == "error":
                    raise VerificationDriftError(
                        f"Verification drift detected: {result.diff}"
                    )

        return live_response

    def _compute_request_hash(self, request: dict[str, Any]) -> str:
        """Compute hash for request matching."""
        canonical = canonical_json(request)
        return self._payload_store.compute_hash(canonical.encode())
```

### Step 3: Run tests

Run: `pytest tests/core/calls/test_wrapper.py -v`
Expected: PASS

---

## Task 7: Redaction Profile Configuration

**Context:** Add configurable redaction profiles for PII and sensitive data in payloads.

**Files:**
- Modify: `src/elspeth/core/config.py` (add RedactionProfile)
- Create: `src/elspeth/core/redaction/__init__.py`
- Create: `src/elspeth/core/redaction/redactor.py`
- Create: `tests/core/redaction/test_redactor.py`

### Step 1: Write the failing test

```python
# tests/core/redaction/test_redactor.py
"""Tests for redaction."""

import pytest


class TestRedactor:
    """Tests for payload redaction."""

    def test_redact_by_field_name(self) -> None:
        """Redacts fields by name pattern."""
        from elspeth.core.redaction import Redactor, RedactionProfile

        profile = RedactionProfile(
            field_patterns=["*password*", "*secret*", "*token*"]
        )
        redactor = Redactor(profile)

        data = {
            "username": "john",
            "password": "hunter2",
            "api_token": "abc123",
            "data": "visible",
        }

        redacted = redactor.redact(data)

        assert redacted["username"] == "john"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["api_token"] == "[REDACTED]"
        assert redacted["data"] == "visible"

    def test_redact_nested_fields(self) -> None:
        """Redacts nested fields."""
        from elspeth.core.redaction import Redactor, RedactionProfile

        profile = RedactionProfile(field_patterns=["*password*"])
        redactor = Redactor(profile)

        data = {
            "user": {
                "name": "john",
                "password": "secret",
            }
        }

        redacted = redactor.redact(data)

        assert redacted["user"]["name"] == "john"
        assert redacted["user"]["password"] == "[REDACTED]"

    def test_redact_by_regex(self) -> None:
        """Redacts values matching regex."""
        from elspeth.core.redaction import Redactor, RedactionProfile

        profile = RedactionProfile(
            value_patterns=[r"sk-[a-zA-Z0-9]+"]  # OpenAI key pattern
        )
        redactor = Redactor(profile)

        data = {
            "key": "sk-abc123xyz",
            "model": "gpt-4",
        }

        redacted = redactor.redact(data)

        assert redacted["key"] == "[REDACTED]"
        assert redacted["model"] == "gpt-4"
```

### Step 2: Create Redactor

```python
# src/elspeth/core/redaction/__init__.py
"""Redaction utilities for sensitive data."""

from elspeth.core.redaction.redactor import Redactor, RedactionProfile

__all__ = ["Redactor", "RedactionProfile"]


# src/elspeth/core/redaction/redactor.py
"""Redactor for sensitive data in payloads."""

import re
import fnmatch
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RedactionProfile:
    """Configuration for what to redact."""

    field_patterns: list[str] = field(default_factory=list)  # fnmatch patterns
    value_patterns: list[str] = field(default_factory=list)  # regex patterns
    replacement: str = "[REDACTED]"


class Redactor:
    """Redacts sensitive data from payloads."""

    def __init__(self, profile: RedactionProfile) -> None:
        self._profile = profile
        self._value_regexes = [
            re.compile(p) for p in profile.value_patterns
        ]

    def redact(self, data: Any) -> Any:
        """Redact sensitive data from a payload.

        Args:
            data: Data to redact (dict, list, or primitive)

        Returns:
            Redacted copy of data
        """
        if isinstance(data, dict):
            return self._redact_dict(data)
        elif isinstance(data, list):
            return [self.redact(item) for item in data]
        else:
            return self._redact_value(data)

    def _redact_dict(self, data: dict) -> dict:
        """Redact a dictionary."""
        result = {}
        for key, value in data.items():
            if self._should_redact_field(key):
                result[key] = self._profile.replacement
            else:
                result[key] = self.redact(value)
        return result

    def _redact_value(self, value: Any) -> Any:
        """Redact a primitive value if it matches patterns."""
        if not isinstance(value, str):
            return value

        for regex in self._value_regexes:
            if regex.search(value):
                return self._profile.replacement

        return value

    def _should_redact_field(self, field_name: str) -> bool:
        """Check if field name matches redaction patterns."""
        for pattern in self._profile.field_patterns:
            if fnmatch.fnmatch(field_name.lower(), pattern.lower()):
                return True
        return False
```

Add RedactionSettings to config:

```python
# Add to src/elspeth/core/config.py

class RedactionSettings(BaseSettings):
    """Configuration for payload redaction."""

    enabled: bool = True
    field_patterns: list[str] = [
        "*password*", "*secret*", "*token*", "*key*", "*credential*"
    ]
    value_patterns: list[str] = [
        r"sk-[a-zA-Z0-9]+",  # OpenAI keys
        r"Bearer\s+[a-zA-Z0-9._-]+",  # Bearer tokens
    ]
```

### Step 3: Run tests

Run: `pytest tests/core/redaction/test_redactor.py -v`
Expected: PASS

---

## Task 8-14: LLM Plugin Pack and Integration

The remaining tasks cover:

- **Task 8:** LiteLLM wrapper transform
- **Task 9:** LLM response parsing and schema validation
- **Task 10:** HTTP call transform (generic external API)
- **Task 11:** PluginContext integration with ExternalCallWrapper
- **Task 12:** CLI `--mode` flag for replay/verify
- **Task 13:** Integration test - full record/replay cycle
- **Task 14:** Integration test - verify mode with drift detection

These follow the same TDD pattern. See architecture docs for LiteLLM integration details.

---

## Summary

Phase 6 adds external call infrastructure:

| Pillar | Tasks | Key Components |
|--------|-------|----------------|
| **Call Recording** | 1 | `CallRecorder`, `calls` table, PayloadStore integration |
| **Run Modes** | 2, 6 | `RunMode` enum, `ExternalCallWrapper` |
| **Secret Handling** | 3 | `SecretFingerprinter`, HMAC-based fingerprints |
| **Replay Mode** | 4 | `CallReplayer`, recorded response retrieval |
| **Verify Mode** | 5 | `CallVerifier`, DeepDiff comparison, drift recording |
| **Redaction** | 7 | `Redactor`, `RedactionProfile`, PII handling |
| **LLM Pack** | 8-9 | LiteLLM wrapper, response parsing |
| **HTTP Plugin** | 10 | Generic HTTP call transform |
| **Integration** | 11-14 | CLI, end-to-end tests |

**Key invariants:**
- Secrets NEVER stored - only HMAC fingerprints
- Every external call recorded with request/response hashes
- Payloads stored in PayloadStore, hashes in Landscape
- Redaction happens BEFORE storage

**New CLI flags:**
- `--mode live|replay|verify`
- `--replay-from <run_id>`
