# Phase 6: External Calls (Tasks 1-14)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add infrastructure for recording, replaying, and verifying external calls (LLM, HTTP, etc.). This enables reproducibility and audit compliance for non-deterministic operations.

**Architecture:** External calls are recorded in the `calls` table with request/response hashes and PayloadStore refs. Run modes (live/replay/verify) control whether calls execute live, use recorded responses, or compare both.

**Tech Stack:** Python 3.11+, LiteLLM (LLM providers), DeepDiff (verification), httpx (HTTP calls)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.payload_store`
- Phase 3A: `elspeth.core.landscape` (schema includes `calls` table)
- Phase 5: `elspeth.core.rate_limit` (for rate limiting external calls)

**Phase 5 Integration Note:**
The rate_limit module from Phase 5 should wrap external calls in ExternalCallWrapper. The integration point is:
1. ExternalCallWrapper._execute_live() should check rate limits before calling executor()
2. Use `rate_limit.acquire(call_type)` before execution
3. If rate limited, raise `RateLimitExceededError` instead of recording a failed call
4. This prevents wasted calls to rate-limited APIs and provides clear failure signals

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

## Schema Pre-requisite

Before implementing Phase 6 tasks, add the following index to `src/elspeth/core/landscape/schema.py`:

```python
# Add after ix_calls_state index definition
Index("ix_calls_request_hash", calls_table.c.request_hash)
```

**Rationale:** Replay mode queries calls by `request_hash` to find matching recorded responses. Without this index, replay lookups require a full table scan which degrades performance significantly for large datasets.

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
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def node_state_id(self, landscape_db):
        """Create a node_state to attach calls to."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table, node_states_table
        )

        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="run-001", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="running"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node-001", run_id="run-001", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id="run-001", source_node_id="node-001",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-001", row_id="row-001", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="state-001", token_id="tok-001", node_id="node-001",
                step_index=0, attempt=0, status="running",
                input_hash="input-hash", started_at=now
            ))
            conn.commit()
        return "state-001"

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

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from elspeth.core.canonical import canonical_json


def _compute_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


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
        request_bytes = request_canonical.encode()
        request_hash = _compute_hash(request_bytes)
        request_ref = self._payload_store.store(request_bytes)

        response_hash = None
        response_ref = None
        if response is not None:
            response_canonical = canonical_json(response)
            response_bytes = response_canonical.encode()
            response_hash = _compute_hash(response_bytes)
            response_ref = self._payload_store.store(response_bytes)

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

from pydantic import BaseModel, ConfigDict, model_validator


class RunMode(Enum):
    """Run mode for external calls."""
    LIVE = "live"        # Execute live, record request/response
    REPLAY = "replay"    # Use recorded responses, no live calls
    VERIFY = "verify"    # Execute live AND compare to recorded


class ExternalCallSettings(BaseModel):
    """Configuration for external call handling."""

    model_config = ConfigDict(frozen=True)

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

    def test_verify_fingerprint_matches_correct_secret(self) -> None:
        """verify_fingerprint returns True for matching secret."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-key")

        secret = "my-api-key"
        fp = fingerprinter.fingerprint(secret)

        # Verification should pass
        assert fingerprinter.verify_fingerprint(secret, fp) is True

    def test_verify_fingerprint_rejects_wrong_secret(self) -> None:
        """verify_fingerprint returns False for wrong secret."""
        from elspeth.core.secrets import SecretFingerprinter

        fingerprinter = SecretFingerprinter(key=b"test-key")

        fp = fingerprinter.fingerprint("correct-secret")

        # Verification with wrong secret should fail
        assert fingerprinter.verify_fingerprint("wrong-secret", fp) is False
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

import hashlib
import pytest
from datetime import datetime, timezone


def _compute_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


class TestCallReplayer:
    """Tests for replaying recorded calls."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def replayer(self, landscape_db, payload_store):
        """Create CallReplayer for tests."""
        from elspeth.core.calls import CallReplayer
        return CallReplayer(landscape_db, payload_store)

    @pytest.fixture
    def recorded_call_fixture(self, landscape_db, payload_store):
        """Create a recorded call to replay from."""
        import json
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table,
            node_states_table, calls_table
        )
        from elspeth.core.canonical import canonical_json

        now = datetime.now(timezone.utc)

        # Set up run and node hierarchy
        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="source-run-001", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node-001", run_id="source-run-001", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id="source-run-001", source_node_id="node-001",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-001", row_id="row-001", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="state-001", token_id="tok-001", node_id="node-001",
                step_index=0, attempt=0, status="completed",
                input_hash="input-hash", started_at=now
            ))

            # Create recorded call with known request hash
            request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
            response = {"choices": [{"message": {"content": "Hi there!"}}]}

            request_canonical = canonical_json(request)
            response_canonical = canonical_json(response)
            request_hash = _compute_hash(request_canonical.encode())

            request_ref = payload_store.store(request_canonical.encode())
            response_ref = payload_store.store(response_canonical.encode())

            conn.execute(calls_table.insert().values(
                call_id="call-001", state_id="state-001", call_index=0,
                call_type="llm", status="success",
                request_hash=request_hash, request_ref=request_ref,
                response_hash=_compute_hash(response_canonical.encode()),
                response_ref=response_ref, latency_ms=100.0, created_at=now
            ))
            conn.commit()

        class RecordedCallInfo:
            run_id = "source-run-001"
            request_hash = request_hash
            state_id = "state-001"

        return RecordedCallInfo()

    @pytest.fixture
    def state_with_multiple_calls(self, landscape_db, payload_store):
        """Create a state with multiple recorded calls."""
        import json
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table,
            node_states_table, calls_table
        )
        from elspeth.core.canonical import canonical_json

        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="multi-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="multi-node", run_id="multi-run", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="multi-row", run_id="multi-run", source_node_id="multi-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="multi-tok", row_id="multi-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="multi-state", token_id="multi-tok", node_id="multi-node",
                step_index=0, attempt=0, status="completed",
                input_hash="input-hash", started_at=now
            ))

            # Create multiple calls
            for i in range(3):
                response = {"result": f"response-{i}"}
                response_canonical = canonical_json(response)
                response_ref = payload_store.store(response_canonical.encode())

                conn.execute(calls_table.insert().values(
                    call_id=f"multi-call-{i}", state_id="multi-state", call_index=i,
                    call_type="http", status="success",
                    request_hash=f"req-hash-{i}", request_ref=f"req-ref-{i}",
                    response_hash=_compute_hash(response_canonical.encode()),
                    response_ref=response_ref, latency_ms=50.0, created_at=now
                ))
            conn.commit()

        return "multi-state"

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
        try:
            payload = self._payload_store.retrieve(result.response_ref)
        except KeyError:
            # Payload was purged or missing
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

        try:
            payload = self._payload_store.retrieve(result.response_ref)
        except KeyError:
            # Payload was purged or missing
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
from datetime import datetime, timezone


class TestCallVerifier:
    """Tests for verify mode."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def verifier(self, landscape_db, payload_store):
        """Create CallVerifier for tests."""
        from elspeth.core.calls import CallVerifier
        return CallVerifier(landscape_db, payload_store)

    @pytest.fixture
    def call_id(self, landscape_db):
        """Create a call record for drift testing."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table,
            node_states_table, calls_table
        )

        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="verify-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="running"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="verify-node", run_id="verify-run", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="verify-row", run_id="verify-run", source_node_id="verify-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="verify-tok", row_id="verify-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="verify-state", token_id="verify-tok", node_id="verify-node",
                step_index=0, attempt=0, status="running",
                input_hash="input-hash", started_at=now
            ))
            conn.execute(calls_table.insert().values(
                call_id="call-for-drift", state_id="verify-state", call_index=0,
                call_type="llm", status="success",
                request_hash="req-hash", latency_ms=100.0, created_at=now
            ))
            conn.commit()

        return "call-for-drift"

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

import hashlib
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock


def _compute_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


class TestExternalCallWrapper:
    """Tests for mode-aware call execution."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def node_state_id(self, landscape_db):
        """Create a node_state for the wrapper."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table, node_states_table
        )

        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="wrapper-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="running"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="wrapper-node", run_id="wrapper-run", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="wrapper-row", run_id="wrapper-run", source_node_id="wrapper-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="wrapper-tok", row_id="wrapper-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="wrapper-state", token_id="wrapper-tok", node_id="wrapper-node",
                step_index=0, attempt=0, status="running",
                input_hash="input-hash", started_at=now
            ))
            conn.commit()
        return "wrapper-state"

    @pytest.fixture
    def source_run_with_recording(self, landscape_db, payload_store):
        """Create a source run with recorded calls for replay/verify."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table,
            node_states_table, calls_table
        )
        from elspeth.core.canonical import canonical_json

        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="source-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="source-node", run_id="source-run", plugin_name="llm_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="source-row", run_id="source-run", source_node_id="source-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="source-tok", row_id="source-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="source-state", token_id="source-tok", node_id="source-node",
                step_index=0, attempt=0, status="completed",
                input_hash="input-hash", started_at=now
            ))

            # Create recorded call with known request hash
            request = {"query": "test"}
            response = {"result": "recorded-data"}

            request_canonical = canonical_json(request)
            response_canonical = canonical_json(response)
            request_hash = _compute_hash(request_canonical.encode())

            request_ref = payload_store.store(request_canonical.encode())
            response_ref = payload_store.store(response_canonical.encode())

            conn.execute(calls_table.insert().values(
                call_id="source-call", state_id="source-state", call_index=0,
                call_type="http", status="success",
                request_hash=request_hash, request_ref=request_ref,
                response_hash=_compute_hash(response_canonical.encode()),
                response_ref=response_ref, latency_ms=100.0, created_at=now
            ))
            conn.commit()

        return "source-run"

    @pytest.fixture
    def wrapper_live_mode(self, landscape_db, payload_store, node_state_id):
        """Create wrapper in LIVE mode."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        recorder = CallRecorder(landscape_db, payload_store)

        return ExternalCallWrapper(
            mode=RunMode.LIVE,
            recorder=recorder,
            replayer=None,
            verifier=None,
            payload_store=payload_store,
            state_id=node_state_id,
        )

    @pytest.fixture
    def wrapper_replay_mode(self, landscape_db, payload_store, node_state_id, source_run_with_recording):
        """Create wrapper in REPLAY mode with recordings."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        recorder = CallRecorder(landscape_db, payload_store)
        replayer = CallReplayer(landscape_db, payload_store)

        return ExternalCallWrapper(
            mode=RunMode.REPLAY,
            recorder=recorder,
            replayer=replayer,
            verifier=None,
            payload_store=payload_store,
            state_id=node_state_id,
            source_run_id=source_run_with_recording,
        )

    @pytest.fixture
    def wrapper_verify_mode(self, landscape_db, payload_store, node_state_id, source_run_with_recording):
        """Create wrapper in VERIFY mode."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer, CallVerifier
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        recorder = CallRecorder(landscape_db, payload_store)
        replayer = CallReplayer(landscape_db, payload_store)
        verifier = CallVerifier(landscape_db, payload_store)

        return ExternalCallWrapper(
            mode=RunMode.VERIFY,
            recorder=recorder,
            replayer=replayer,
            verifier=verifier,
            payload_store=payload_store,
            state_id=node_state_id,
            source_run_id=source_run_with_recording,
        )

    @pytest.fixture
    def wrapper_replay_mode_no_recording(self, landscape_db, payload_store, node_state_id):
        """Create wrapper in REPLAY mode without recordings."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        recorder = CallRecorder(landscape_db, payload_store)
        replayer = CallReplayer(landscape_db, payload_store)

        return ExternalCallWrapper(
            mode=RunMode.REPLAY,
            recorder=recorder,
            replayer=replayer,
            verifier=None,
            payload_store=payload_store,
            state_id=node_state_id,
            source_run_id="nonexistent-run",
        )

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

import hashlib
import time
from typing import Any, Callable

from elspeth.core.config import RunMode
from elspeth.core.calls.recorder import CallRecorder
from elspeth.core.calls.replayer import CallReplayer
from elspeth.core.calls.verifier import CallVerifier
from elspeth.core.canonical import canonical_json


def _compute_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


# Exception types that external executors may raise.
# These are the common base types for HTTP, network, and API errors.
#
# NOTE: LiteLLM exceptions (APIError, RateLimitError, etc.) inherit from
# these base types, so they will be caught. If using a provider that raises
# different exception types, add them to this tuple.
ExternalCallErrors = (
    OSError,           # Network errors (ConnectionError, TimeoutError are subclasses)
    RuntimeError,      # General runtime failures
    ValueError,        # Invalid data/response errors
    LookupError,       # Key/index errors in responses
)


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
            response, _call_id = self._execute_live(executor, request, call_type)
            return response

    def _execute_live(
        self,
        executor: Callable,
        request: dict[str, Any],
        call_type: str,
    ) -> tuple[dict[str, Any], str]:
        """Execute live and record.

        Returns:
            Tuple of (response, call_id) for drift recording.
        """
        start = time.monotonic()
        error = None
        response = None
        call_id = None

        try:
            response = executor()
        except ExternalCallErrors as e:
            # Record error metadata for audit before re-raising
            error = {"type": type(e).__name__, "message": str(e)}
            raise
        finally:
            latency = (time.monotonic() - start) * 1000

            recorded = self._recorder.record_call(
                state_id=self._state_id,
                call_index=self._call_index,
                call_type=call_type,
                request=request,
                response=response,
                latency_ms=latency,
                error=error,
            )
            call_id = recorded.call_id
            self._call_index += 1

        return response, call_id

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

        # Execute live - get both response and call_id for drift recording
        live_response, call_id = self._execute_live(executor, request, call_type)

        # Compare if we have recorded
        if recorded is not None and self._verifier is not None:
            result = self._verifier.verify(recorded, live_response)

            if not result.matched:
                # Record drift in Landscape for audit trail
                self._verifier.record_drift(call_id, result)

                if self._fail_on_drift and result.severity == "error":
                    raise VerificationDriftError(
                        f"Verification drift detected: {result.diff}"
                    )

        return live_response

    def _compute_request_hash(self, request: dict[str, Any]) -> str:
        """Compute hash for request matching."""
        canonical = canonical_json(request)
        return _compute_hash(canonical.encode())
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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RedactionProfile(BaseModel):
    """Configuration for what to redact."""

    model_config = ConfigDict(frozen=True)

    field_patterns: list[str] = Field(default_factory=list)  # fnmatch patterns
    value_patterns: list[str] = Field(default_factory=list)  # regex patterns
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

class RedactionSettings(BaseModel):
    """Configuration for payload redaction."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    field_patterns: list[str] = Field(default=[
        "*password*", "*secret*", "*token*", "*key*", "*credential*"
    ])
    value_patterns: list[str] = Field(default=[
        r"sk-[a-zA-Z0-9]+",  # OpenAI keys
        r"Bearer\s+[a-zA-Z0-9._-]+",  # Bearer tokens
    ])
```

### Step 3: Run tests

Run: `pytest tests/core/redaction/test_redactor.py -v`
Expected: PASS

---

## Task 8: LiteLLM Wrapper Transform

**Context:** Create a transform that wraps LiteLLM for unified LLM access with full audit recording.

**Files:**
- Create: `src/elspeth/plugins/transforms/llm_transform.py`
- Create: `tests/plugins/transforms/test_llm_transform.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/test_llm_transform.py
"""Tests for LLM Transform using LiteLLM."""

import pytest
from unittest.mock import Mock, patch


class TestLLMTransform:
    """Tests for LLM transform wrapper."""

    def test_transform_makes_llm_call(self) -> None:
        """Transform calls LiteLLM with correct params."""
        from elspeth.plugins.transforms.llm_transform import LLMTransform

        transform = LLMTransform(
            model="gpt-4",
            temperature=0.7,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[Mock(message=Mock(content="Hello!"))]
            )

            result = transform.process(
                context=Mock(),
                row={"prompt": "Say hello"},
            )

        mock_completion.assert_called_once()
        assert result["response"] == "Hello!"

    def test_transform_records_call_via_wrapper(self) -> None:
        """Transform uses ExternalCallWrapper for recording."""
        from elspeth.plugins.transforms.llm_transform import LLMTransform
        from elspeth.core.config import RunMode

        mock_wrapper = Mock()
        mock_wrapper.execute.return_value = {"choices": [{"message": {"content": "Hi"}}]}

        transform = LLMTransform(model="gpt-4")
        transform._call_wrapper = mock_wrapper

        result = transform.process(
            context=Mock(),
            row={"prompt": "Hello"},
        )

        mock_wrapper.execute.assert_called_once()
        assert "llm" in str(mock_wrapper.execute.call_args)

    def test_transform_handles_api_error(self) -> None:
        """Transform handles LiteLLM errors gracefully."""
        from elspeth.plugins.transforms.llm_transform import LLMTransform, LLMCallError

        transform = LLMTransform(model="gpt-4")

        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API rate limited")

            with pytest.raises(LLMCallError) as exc_info:
                transform.process(
                    context=Mock(),
                    row={"prompt": "Test"},
                )

        assert "rate limited" in str(exc_info.value)
```

### Step 2: Create LLMTransform

```python
# src/elspeth/plugins/transforms/llm_transform.py
"""LLM Transform using LiteLLM."""

from typing import Any

import litellm

from elspeth.core.config import RunMode
from elspeth.core.calls.wrapper import ExternalCallWrapper, ExternalCallErrors


class LLMCallError(Exception):
    """Error during LLM call."""
    pass


class LLMTransform:
    """Transform that calls LLM via LiteLLM.

    Uses ExternalCallWrapper for recording, replay, and verification.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_wrapper: ExternalCallWrapper | None = None

    def set_call_wrapper(self, wrapper: ExternalCallWrapper) -> None:
        """Set the call wrapper for recording/replay."""
        self._call_wrapper = wrapper

    def process(self, context: Any, row: dict[str, Any]) -> dict[str, Any]:
        """Process a row through LLM.

        Args:
            context: Plugin context
            row: Row data with 'prompt' field

        Returns:
            Row with 'response' field added
        """
        prompt = row["prompt"]  # Required field - fail loudly if missing

        request = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        if self._max_tokens:
            request["max_tokens"] = self._max_tokens

        try:
            if self._call_wrapper:
                response = self._call_wrapper.execute(
                    executor=lambda: self._call_litellm(request),
                    request=request,
                    call_type="llm",
                )
            else:
                response = self._call_litellm(request)
        except ExternalCallErrors as e:
            raise LLMCallError(f"LLM call failed: {e}") from e

        content = response["choices"][0]["message"]["content"]

        return {**row, "response": content}

    def _call_litellm(self, request: dict[str, Any]) -> dict[str, Any]:
        """Make the actual LiteLLM call."""
        response = litellm.completion(**request)
        return response.model_dump()
```

### Step 3: Run tests

Run: `pytest tests/plugins/transforms/test_llm_transform.py -v`
Expected: PASS

---

## Task 9: LLM Response Parsing and Validation

**Context:** Add response parsing utilities for structured LLM output with schema validation.

**Files:**
- Create: `src/elspeth/plugins/transforms/llm_parser.py`
- Create: `tests/plugins/transforms/test_llm_parser.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/test_llm_parser.py
"""Tests for LLM response parsing."""

import pytest


class TestLLMParser:
    """Tests for structured LLM response parsing."""

    def test_parse_json_response(self) -> None:
        """Parses JSON from LLM response."""
        from elspeth.plugins.transforms.llm_parser import LLMParser

        parser = LLMParser()

        response = '{"name": "John", "age": 30}'
        result = parser.parse_json(response)

        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_markdown_wrapper(self) -> None:
        """Handles JSON wrapped in markdown code blocks."""
        from elspeth.plugins.transforms.llm_parser import LLMParser

        parser = LLMParser()

        response = '''```json
{"name": "John"}
```'''
        result = parser.parse_json(response)

        assert result == {"name": "John"}

    def test_validate_against_schema(self) -> None:
        """Validates parsed response against Pydantic schema."""
        from pydantic import BaseModel
        from elspeth.plugins.transforms.llm_parser import LLMParser

        class PersonSchema(BaseModel):
            name: str
            age: int

        parser = LLMParser()

        response = '{"name": "John", "age": 30}'
        result = parser.parse_and_validate(response, PersonSchema)

        assert isinstance(result, PersonSchema)
        assert result.name == "John"

    def test_validation_fails_for_invalid_data(self) -> None:
        """Raises error when response doesn't match schema."""
        from pydantic import BaseModel, ValidationError
        from elspeth.plugins.transforms.llm_parser import LLMParser, LLMParseError

        class PersonSchema(BaseModel):
            name: str
            age: int

        parser = LLMParser()

        response = '{"name": "John"}'  # Missing age

        with pytest.raises(LLMParseError):
            parser.parse_and_validate(response, PersonSchema)
```

### Step 2: Create LLMParser

```python
# src/elspeth/plugins/transforms/llm_parser.py
"""LLM response parsing utilities."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


class LLMParseError(Exception):
    """Error parsing LLM response."""
    pass


T = TypeVar("T", bound=BaseModel)


class LLMParser:
    """Parses and validates structured LLM responses."""

    def parse_json(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response.

        Handles:
        - Raw JSON
        - JSON wrapped in markdown code blocks

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict

        Raises:
            LLMParseError: If JSON parsing fails
        """
        # Try to extract from markdown code block
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if code_block:
            response = code_block.group(1)

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            raise LLMParseError(f"Failed to parse JSON: {e}") from e

    def parse_and_validate(self, response: str, schema: type[T]) -> T:
        """Parse JSON and validate against Pydantic schema.

        Args:
            response: LLM response text
            schema: Pydantic model class

        Returns:
            Validated model instance

        Raises:
            LLMParseError: If parsing or validation fails
        """
        data = self.parse_json(response)

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise LLMParseError(f"Schema validation failed: {e}") from e
```

### Step 3: Run tests

Run: `pytest tests/plugins/transforms/test_llm_parser.py -v`
Expected: PASS

---

## Task 10: HTTP Call Transform

**Context:** Create a generic HTTP call transform for external API integration.

**Files:**
- Create: `src/elspeth/plugins/transforms/http_transform.py`
- Create: `tests/plugins/transforms/test_http_transform.py`

### Step 1: Write the failing test

```python
# tests/plugins/transforms/test_http_transform.py
"""Tests for HTTP Transform."""

import pytest
from unittest.mock import Mock, patch


class TestHTTPTransform:
    """Tests for HTTP call transform."""

    def test_get_request(self) -> None:
        """Makes GET request and returns response."""
        from elspeth.plugins.transforms.http_transform import HTTPTransform

        transform = HTTPTransform(
            method="GET",
            url_template="https://api.example.com/users/{user_id}",
        )

        with patch("httpx.request") as mock_request:
            mock_request.return_value = Mock(
                status_code=200,
                json=lambda: {"name": "John"},
            )

            result = transform.process(
                context=Mock(),
                row={"user_id": "123"},
            )

        assert result["http_response"] == {"name": "John"}

    def test_post_request_with_body(self) -> None:
        """Makes POST request with JSON body."""
        from elspeth.plugins.transforms.http_transform import HTTPTransform

        transform = HTTPTransform(
            method="POST",
            url_template="https://api.example.com/users",
            body_template={"name": "{name}", "email": "{email}"},
        )

        with patch("httpx.request") as mock_request:
            mock_request.return_value = Mock(
                status_code=201,
                json=lambda: {"id": "new-user-123"},
            )

            result = transform.process(
                context=Mock(),
                row={"name": "John", "email": "john@example.com"},
            )

        assert result["http_response"]["id"] == "new-user-123"

    def test_uses_call_wrapper_for_recording(self) -> None:
        """Uses ExternalCallWrapper for audit recording."""
        from elspeth.plugins.transforms.http_transform import HTTPTransform

        mock_wrapper = Mock()
        mock_wrapper.execute.return_value = {"data": "test"}

        transform = HTTPTransform(
            method="GET",
            url_template="https://api.example.com/test",
        )
        transform._call_wrapper = mock_wrapper

        result = transform.process(
            context=Mock(),
            row={},
        )

        mock_wrapper.execute.assert_called_once()
```

### Step 2: Create HTTPTransform

```python
# src/elspeth/plugins/transforms/http_transform.py
"""HTTP Call Transform."""

from typing import Any

import httpx

from elspeth.core.calls.wrapper import ExternalCallWrapper, ExternalCallErrors


class HTTPCallError(Exception):
    """Error during HTTP call."""
    pass


class HTTPTransform:
    """Transform that makes HTTP requests.

    Supports URL and body templating from row fields.
    """

    def __init__(
        self,
        method: str,
        url_template: str,
        body_template: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._method = method.upper()
        self._url_template = url_template
        self._body_template = body_template
        self._headers = headers or {}
        self._timeout = timeout
        self._call_wrapper: ExternalCallWrapper | None = None

    def set_call_wrapper(self, wrapper: ExternalCallWrapper) -> None:
        """Set the call wrapper for recording/replay."""
        self._call_wrapper = wrapper

    def process(self, context: Any, row: dict[str, Any]) -> dict[str, Any]:
        """Process row through HTTP call.

        Args:
            context: Plugin context
            row: Row data for templating

        Returns:
            Row with 'http_response' added
        """
        url = self._url_template.format(**row)

        body = None
        if self._body_template:
            body = self._render_template(self._body_template, row)

        request = {
            "method": self._method,
            "url": url,
            "headers": self._headers,
        }
        if body:
            request["json"] = body

        try:
            if self._call_wrapper:
                response = self._call_wrapper.execute(
                    executor=lambda: self._make_request(request),
                    request=request,
                    call_type="http",
                )
            else:
                response = self._make_request(request)
        except ExternalCallErrors as e:
            raise HTTPCallError(f"HTTP call failed: {e}") from e

        return {**row, "http_response": response}

    def _make_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Make the actual HTTP request."""
        response = httpx.request(
            method=request["method"],
            url=request["url"],
            headers=request.get("headers"),
            json=request.get("json"),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def _render_template(self, template: dict, row: dict) -> dict:
        """Render template with row values."""
        result = {}
        for key, value in template.items():
            if isinstance(value, str):
                result[key] = value.format(**row)
            elif isinstance(value, dict):
                result[key] = self._render_template(value, row)
            else:
                result[key] = value
        return result
```

### Step 3: Run tests

Run: `pytest tests/plugins/transforms/test_http_transform.py -v`
Expected: PASS

---

## Task 11: PluginContext Integration

**Context:** Integrate ExternalCallWrapper into PluginContext so transforms automatically get call recording.

**Files:**
- Modify: `src/elspeth/engine/plugin_context.py` (add call wrapper factory)
- Modify: `tests/engine/test_plugin_context.py` (add integration tests)

### Step 1: Write the failing test

```python
# Add to tests/engine/test_plugin_context.py

from unittest.mock import Mock


class TestPluginContextCallWrapper:
    """Tests for call wrapper integration."""

    def test_context_creates_call_wrapper(self) -> None:
        """PluginContext creates ExternalCallWrapper for transforms."""
        from elspeth.engine.plugin_context import PluginContext
        from elspeth.core.config import RunMode

        context = PluginContext(
            run_mode=RunMode.LIVE,
            landscape_db=Mock(),
            payload_store=Mock(),
            state_id="state-001",
        )

        wrapper = context.get_call_wrapper()

        assert wrapper is not None
        assert wrapper._mode == RunMode.LIVE

    def test_replay_mode_configures_replayer(self) -> None:
        """Replay mode creates wrapper with replayer configured."""
        from elspeth.engine.plugin_context import PluginContext
        from elspeth.core.config import RunMode

        context = PluginContext(
            run_mode=RunMode.REPLAY,
            landscape_db=Mock(),
            payload_store=Mock(),
            state_id="state-001",
            source_run_id="source-run-123",
        )

        wrapper = context.get_call_wrapper()

        assert wrapper._replayer is not None
        assert wrapper._source_run_id == "source-run-123"
```

### Step 2: Implement PluginContext integration

```python
# Add to src/elspeth/engine/plugin_context.py

from elspeth.core.config import RunMode
from elspeth.core.calls.wrapper import ExternalCallWrapper
from elspeth.core.calls import CallRecorder, CallReplayer, CallVerifier


class PluginContext:
    """Context provided to plugins during execution."""

    def __init__(
        self,
        run_mode: RunMode,
        landscape_db,
        payload_store,
        state_id: str,
        source_run_id: str | None = None,
        fail_on_drift: bool = True,
    ) -> None:
        self._run_mode = run_mode
        self._landscape_db = landscape_db
        self._payload_store = payload_store
        self._state_id = state_id
        self._source_run_id = source_run_id
        self._fail_on_drift = fail_on_drift
        self._call_wrapper: ExternalCallWrapper | None = None

    def get_call_wrapper(self) -> ExternalCallWrapper:
        """Get or create the ExternalCallWrapper for this context.

        Returns:
            Configured wrapper for the current run mode
        """
        if self._call_wrapper is None:
            recorder = CallRecorder(self._landscape_db, self._payload_store)

            replayer = None
            verifier = None

            if self._run_mode in (RunMode.REPLAY, RunMode.VERIFY):
                replayer = CallReplayer(self._landscape_db, self._payload_store)

            if self._run_mode == RunMode.VERIFY:
                verifier = CallVerifier(self._landscape_db, self._payload_store)

            self._call_wrapper = ExternalCallWrapper(
                mode=self._run_mode,
                recorder=recorder,
                replayer=replayer,
                verifier=verifier,
                payload_store=self._payload_store,
                state_id=self._state_id,
                source_run_id=self._source_run_id,
                fail_on_drift=self._fail_on_drift,
            )

        return self._call_wrapper
```

### Step 3: Run tests

Run: `pytest tests/engine/test_plugin_context.py -v`
Expected: PASS

---

## Task 12: CLI --mode Flag

**Context:** Add CLI flags for run mode control.

**Files:**
- Modify: `src/elspeth/cli.py` (add --mode and --replay-from flags)
- Modify: `tests/test_cli.py` (add CLI flag tests)

### Step 1: Write the failing test

```python
# Add to tests/test_cli.py

import pytest
from typer.testing import CliRunner


class TestCLIRunMode:
    """Tests for CLI run mode flags."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        from elspeth.cli import app
        runner = CliRunner()
        return lambda args: runner.invoke(app, args)

    def test_default_mode_is_live(self, cli_runner) -> None:
        """Default mode is live."""
        result = cli_runner(["run", "settings.yaml"])

        # Should use live mode by default
        assert "mode=live" in result.output or result.exit_code == 0

    def test_replay_mode_requires_source_run(self, cli_runner) -> None:
        """--mode replay requires --replay-from."""
        result = cli_runner(["run", "--mode", "replay", "settings.yaml"])

        assert result.exit_code != 0
        assert "replay-from" in result.output.lower()

    def test_verify_mode_requires_source_run(self, cli_runner) -> None:
        """--mode verify requires --replay-from."""
        result = cli_runner(["run", "--mode", "verify", "settings.yaml"])

        assert result.exit_code != 0
        assert "replay-from" in result.output.lower()

    def test_replay_mode_with_source_run(self, cli_runner) -> None:
        """--mode replay with --replay-from succeeds."""
        result = cli_runner([
            "run",
            "--mode", "replay",
            "--replay-from", "run-123",
            "settings.yaml",
        ])

        # Should proceed with replay mode
        assert result.exit_code == 0 or "replay" in result.output.lower()
```

### Step 2: Add CLI flags

```python
# Add to src/elspeth/cli.py

import typer
from elspeth.core.config import RunMode

app = typer.Typer()


@app.command()
def run(
    settings_path: str = typer.Argument(..., help="Path to settings YAML"),
    mode: str = typer.Option("live", help="Run mode: live, replay, or verify"),
    replay_from: str | None = typer.Option(
        None, "--replay-from",
        help="Source run ID for replay/verify modes"
    ),
) -> None:
    """Run an ELSPETH pipeline."""
    # Validate mode
    try:
        run_mode = RunMode(mode)
    except ValueError:
        typer.echo(f"Invalid mode: {mode}. Must be: live, replay, verify")
        raise typer.Exit(1)

    # Validate replay-from requirement
    if run_mode in (RunMode.REPLAY, RunMode.VERIFY) and not replay_from:
        typer.echo(f"--replay-from required when mode={mode}")
        raise typer.Exit(1)

    typer.echo(f"Running in {run_mode.value} mode")

    if replay_from:
        typer.echo(f"Source run: {replay_from}")

    # Load settings and run pipeline...
```

### Step 3: Run tests

Run: `pytest tests/test_cli.py::TestCLIRunMode -v`
Expected: PASS

---

## Task 13: Integration Test - Record/Replay Cycle

**Context:** End-to-end test for recording and replaying external calls.

**Files:**
- Create: `tests/integration/test_record_replay.py`

### Step 1: Integration test

```python
# tests/integration/test_record_replay.py
"""Integration tests for record/replay cycle."""

import pytest
from unittest.mock import Mock, patch


class TestRecordReplayCycle:
    """End-to-end tests for external call recording and replay."""

    @pytest.fixture
    def setup_landscape(self, tmp_path):
        """Set up Landscape database for testing."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        return db, payload_store

    def test_record_then_replay(self, setup_landscape) -> None:
        """Can record a call and replay it."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        db, payload_store = setup_landscape

        # Set up database fixtures for the recording phase
        from datetime import datetime, timezone
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table, node_states_table
        )

        now = datetime.now(timezone.utc)
        with db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id="run-001", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="running"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node-001", run_id="run-001", plugin_name="http_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id="run-001", source_node_id="node-001",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-001", row_id="row-001", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="state-001", token_id="tok-001", node_id="node-001",
                step_index=0, attempt=0, status="running",
                input_hash="input-hash", started_at=now
            ))
            conn.commit()

        # Phase 1: Record
        recorder = CallRecorder(db, payload_store)
        record_wrapper = ExternalCallWrapper(
            mode=RunMode.LIVE,
            recorder=recorder,
            replayer=None,
            verifier=None,
            payload_store=payload_store,
            state_id="state-001",
        )

        # Make a call and record it
        mock_executor = Mock(return_value={"result": "live-data"})
        request = {"query": "test-query"}

        live_result = record_wrapper.execute(
            executor=mock_executor,
            request=request,
            call_type="http",
        )

        assert live_result == {"result": "live-data"}
        mock_executor.assert_called_once()

        # Phase 2: Replay
        replayer = CallReplayer(db, payload_store)
        replay_wrapper = ExternalCallWrapper(
            mode=RunMode.REPLAY,
            recorder=recorder,
            replayer=replayer,
            verifier=None,
            payload_store=payload_store,
            state_id="state-002",
            source_run_id="run-001",  # Source from recording
        )

        mock_executor_replay = Mock()

        # Replay should return recorded data without calling executor
        replay_result = replay_wrapper.execute(
            executor=mock_executor_replay,
            request=request,
            call_type="http",
        )

        mock_executor_replay.assert_not_called()
        assert replay_result == {"result": "live-data"}
```

---

## Task 14: Integration Test - Verify Mode

**Context:** End-to-end test for verify mode with drift detection.

**Files:**
- Create: `tests/integration/test_verify_mode.py`

### Step 1: Integration test

```python
# tests/integration/test_verify_mode.py
"""Integration tests for verify mode drift detection."""

import pytest
from unittest.mock import Mock


class TestVerifyMode:
    """End-to-end tests for verify mode."""

    @pytest.fixture
    def setup_with_recorded_calls(self, tmp_path):
        """Set up database with recorded calls for verification."""
        import hashlib
        from datetime import datetime, timezone
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore
        from elspeth.core.calls import CallRecorder
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table,
            node_states_table, calls_table
        )
        from elspeth.core.canonical import canonical_json

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        # Set up complete recorded call fixture
        now = datetime.now(timezone.utc)

        def compute_hash(content: bytes) -> str:
            return hashlib.sha256(content).hexdigest()

        with db.engine.connect() as conn:
            # Source run (what we're verifying against)
            conn.execute(runs_table.insert().values(
                run_id="source-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="source-node", run_id="source-run", plugin_name="http_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="source-row", run_id="source-run", source_node_id="source-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="source-tok", row_id="source-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="source-state", token_id="source-tok", node_id="source-node",
                step_index=0, attempt=0, status="completed",
                input_hash="input-hash", started_at=now
            ))

            # Create recorded call with known request/response
            request = {"query": "test"}
            response = {"result": "recorded-data"}

            request_canonical = canonical_json(request)
            response_canonical = canonical_json(response)
            request_hash = compute_hash(request_canonical.encode())

            request_ref = payload_store.store(request_canonical.encode())
            response_ref = payload_store.store(response_canonical.encode())

            conn.execute(calls_table.insert().values(
                call_id="source-call", state_id="source-state", call_index=0,
                call_type="http", status="success",
                request_hash=request_hash, request_ref=request_ref,
                response_hash=compute_hash(response_canonical.encode()),
                response_ref=response_ref, latency_ms=100.0, created_at=now
            ))

            # Verify run (current run doing verification)
            conn.execute(runs_table.insert().values(
                run_id="verify-run", started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="running"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="verify-node", run_id="verify-run", plugin_name="http_transform",
                node_type="transform", plugin_version="1.0", determinism="external_call",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="verify-row", run_id="verify-run", source_node_id="verify-node",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="verify-tok", row_id="verify-row", created_at=now
            ))
            conn.execute(node_states_table.insert().values(
                state_id="state-verify", token_id="verify-tok", node_id="verify-node",
                step_index=0, attempt=0, status="running",
                input_hash="input-hash", started_at=now
            ))
            conn.commit()

        return db, payload_store

    def test_verify_detects_drift(self, setup_with_recorded_calls) -> None:
        """Verify mode detects when live response differs from recorded."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer, CallVerifier
        from elspeth.core.calls.wrapper import (
            ExternalCallWrapper, VerificationDriftError
        )

        db, payload_store = setup_with_recorded_calls

        verifier = CallVerifier(db, payload_store)
        replayer = CallReplayer(db, payload_store)
        recorder = CallRecorder(db, payload_store)

        wrapper = ExternalCallWrapper(
            mode=RunMode.VERIFY,
            recorder=recorder,
            replayer=replayer,
            verifier=verifier,
            payload_store=payload_store,
            state_id="state-verify",
            source_run_id="source-run",
            fail_on_drift=True,
        )

        # Live response differs from recorded
        mock_executor = Mock(return_value={"result": "different-data"})
        request = {"query": "test"}

        # Should detect drift and raise
        with pytest.raises(VerificationDriftError):
            wrapper.execute(
                executor=mock_executor,
                request=request,
                call_type="http",
            )

    def test_verify_passes_when_matching(self, setup_with_recorded_calls) -> None:
        """Verify mode passes when live matches recorded."""
        from elspeth.core.config import RunMode
        from elspeth.core.calls import CallRecorder, CallReplayer, CallVerifier
        from elspeth.core.calls.wrapper import ExternalCallWrapper

        db, payload_store = setup_with_recorded_calls

        verifier = CallVerifier(db, payload_store)
        replayer = CallReplayer(db, payload_store)
        recorder = CallRecorder(db, payload_store)

        wrapper = ExternalCallWrapper(
            mode=RunMode.VERIFY,
            recorder=recorder,
            replayer=replayer,
            verifier=verifier,
            payload_store=payload_store,
            state_id="state-verify",
            source_run_id="source-run",
            fail_on_drift=True,
        )

        # Live response matches recorded
        mock_executor = Mock(return_value={"result": "recorded-data"})
        request = {"query": "test"}

        # Should pass without error
        result = wrapper.execute(
            executor=mock_executor,
            request=request,
            call_type="http",
        )

        assert result == {"result": "recorded-data"}
```

### Step 3: Run tests

Run: `pytest tests/integration/ -v`
Expected: PASS

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
