# RAG Ingestion Sub-plan 2: ChromaSink Plugin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new sink plugin that writes pipeline rows into a ChromaDB collection, with full audit trail recording, content hashing, and three duplicate-handling modes.

**Architecture:** `ChromaSink` extends `BaseSink` following the established sink pattern (reference: `database_sink.py`). Config extends `DataPluginConfig` and composes `ChromaConnectionConfig` from sub-plan 1. Field mapping is explicit — the operator declares which row fields map to ChromaDB's document/id/metadata concepts. All writes are audited via `ctx.record_call()` with canonical content hashing before the write.

**Tech Stack:** chromadb (>= 0.4), Pydantic v2, ELSPETH plugin infrastructure (BaseSink, DataPluginConfig, SchemaConfig)

**Spec:** `docs/superpowers/specs/2026-03-25-rag-ingestion-pipeline-design.md` (Component 1: ChromaSink Plugin)

**Depends on:** Sub-plan 1 (shared infrastructure) must be merged first.

---

## Review Notes from Sub-plan 1 (read before implementing)

The following changes were made during sub-plan 1's 4-agent review that affect this plan:

1. **`ChromaConnectionConfig` composition pattern.** `ChromaSearchProviderConfig` currently composes `ChromaConnectionConfig` via *validation-by-construction* — it constructs a `ChromaConnectionConfig` in its model_validator and discards it. This was flagged as transitional duplication. When implementing `ChromaSinkConfig`, compose `ChromaConnectionConfig` as a **nested field** rather than duplicating its flat fields. This avoids the field-sync maintenance trap. If the sink config needs `ephemeral` mode, handle it as a separate concern — `ChromaConnectionConfig` deliberately excludes `ephemeral`.

2. **`ChromaSearchProviderConfig.model_config` uses dict literal.** The existing provider config uses `model_config = {"extra": "forbid", "frozen": True}` while `ChromaConnectionConfig` uses `ConfigDict(frozen=True, extra="forbid")`. This inconsistency was noted but deferred. Use `ConfigDict` for the new sink config.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/elspeth/plugins/sinks/chroma_sink.py` | `ChromaSinkConfig` Pydantic model + `ChromaSink` plugin class |
| Create | `tests/unit/plugins/sinks/test_chroma_sink_config.py` | Config validation tests |
| Create | `tests/unit/plugins/sinks/test_chroma_sink.py` | Sink lifecycle and write tests (mocked ChromaDB) |
| Create | `tests/integration/plugins/sinks/test_chroma_sink_pipeline.py` | Full pipeline with real ephemeral ChromaDB + Landscape assertions |

---

### Task 1: `ChromaSinkConfig` — Field Mapping Model

**Files:**
- Create: `tests/unit/plugins/sinks/test_chroma_sink_config.py`
- Create: `src/elspeth/plugins/sinks/chroma_sink.py` (config only)

- [ ] **Step 1: Write config validation tests**

```python
# tests/unit/plugins/sinks/test_chroma_sink_config.py
"""Tests for ChromaSink configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.plugins.sinks.chroma_sink import ChromaSinkConfig


class TestFieldMappingConfig:
    """Tests for field_mapping validation rules."""

    def test_valid_config(self) -> None:
        config = ChromaSinkConfig.from_dict({
            "collection": "science-facts",
            "mode": "persistent",
            "persist_directory": "./chroma_data",
            "distance_function": "cosine",
            "field_mapping": {
                "document": "text_content",
                "id": "doc_id",
                "metadata": ["topic", "subtopic"],
            },
            "on_duplicate": "overwrite",
            "schema": {
                "mode": "fixed",
                "fields": [
                    "doc_id: str",
                    "text_content: str",
                    "topic: str",
                    "subtopic: str",
                ],
            },
        })
        assert config.collection == "science-facts"
        assert config.field_mapping.document == "text_content"
        assert config.field_mapping.id == "doc_id"
        assert config.field_mapping.metadata == ["topic", "subtopic"]

    def test_field_mapping_required(self) -> None:
        with pytest.raises(Exception, match="field_mapping"):
            ChromaSinkConfig.from_dict({
                "collection": "test",
                "mode": "persistent",
                "persist_directory": "./data",
                "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
            })

    def test_on_duplicate_default_is_overwrite(self) -> None:
        config = ChromaSinkConfig.from_dict({
            "collection": "test",
            "mode": "persistent",
            "persist_directory": "./data",
            "field_mapping": {"document": "text", "id": "id", "metadata": []},
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        })
        assert config.on_duplicate == "overwrite"

    def test_on_duplicate_rejects_invalid_value(self) -> None:
        with pytest.raises(Exception):
            ChromaSinkConfig.from_dict({
                "collection": "test",
                "mode": "persistent",
                "persist_directory": "./data",
                "field_mapping": {"document": "text", "id": "id", "metadata": []},
                "on_duplicate": "invalid",
                "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
            })

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(Exception, match="extra"):
            ChromaSinkConfig.from_dict({
                "collection": "test",
                "mode": "persistent",
                "persist_directory": "./data",
                "field_mapping": {"document": "text", "id": "id", "metadata": []},
                "unknown_extra": "value",
                "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
            })


class TestConnectionValidation:
    """Tests that ChromaConnectionConfig validation flows through."""

    def test_persistent_mode_requires_persist_directory(self) -> None:
        with pytest.raises(Exception, match="persist_directory"):
            ChromaSinkConfig.from_dict({
                "collection": "test",
                "mode": "persistent",
                "field_mapping": {"document": "t", "id": "i", "metadata": []},
                "schema": {"mode": "fixed", "fields": ["i: str", "t: str"]},
            })

    def test_client_mode_requires_host(self) -> None:
        with pytest.raises(Exception, match="host"):
            ChromaSinkConfig.from_dict({
                "collection": "test",
                "mode": "client",
                "field_mapping": {"document": "t", "id": "i", "metadata": []},
                "schema": {"mode": "fixed", "fields": ["i: str", "t: str"]},
            })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ChromaSinkConfig` with field mapping**

```python
# src/elspeth/plugins/sinks/chroma_sink.py
"""ChromaDB vector store sink plugin.

Writes pipeline rows into a ChromaDB collection. Each row becomes a document.
ChromaDB handles embedding internally via its configured embedding function.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)
from elspeth.plugins.infrastructure.config_base import DataPluginConfig


class FieldMappingConfig(BaseModel):
    """Maps row fields to ChromaDB document concepts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: str = Field(description="Row field containing text to embed")
    id: str = Field(description="Row field containing document ID")
    metadata: list[str] = Field(
        default_factory=list,
        description="Row fields to include as ChromaDB metadata",
    )


class ChromaSinkConfig(DataPluginConfig):
    """Configuration for ChromaDB vector store sink.

    Connection fields are validated by ChromaConnectionConfig rules.
    Field mapping is required — no convention-based defaults.
    """

    # Connection fields (inline, not composed — DataPluginConfig.from_dict
    # expects flat config, not nested sub-models)
    collection: str = Field(description="ChromaDB collection name")
    mode: Literal["persistent", "client"] = Field(
        description="Connection mode"
    )
    persist_directory: str | None = Field(default=None)
    host: str | None = Field(default=None)
    port: int = Field(default=8000)
    ssl: bool = Field(default=True)
    distance_function: Literal["cosine", "l2", "ip"] = Field(default="cosine")

    # Sink-specific fields
    field_mapping: FieldMappingConfig = Field(
        description="Maps row fields to ChromaDB document/id/metadata"
    )
    on_duplicate: Literal["overwrite", "skip", "error"] = Field(
        default="overwrite",
        description="Behaviour when a document ID already exists",
    )

    @model_validator(mode="after")
    def validate_connection(self) -> ChromaSinkConfig:
        """Delegate connection validation to ChromaConnectionConfig."""
        ChromaConnectionConfig(
            collection=self.collection,
            mode=self.mode,
            persist_directory=self.persist_directory,
            host=self.host,
            port=self.port,
            ssl=self.ssl,
            distance_function=self.distance_function,
        )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink_config.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_chroma_sink_config.py
git commit -m "feat: add ChromaSinkConfig with field mapping and connection validation"
```

---

### Task 2: `ChromaSink` — Class Skeleton and `on_start()` Lifecycle

**Files:**
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py`
- Create: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1: Write tests for `on_start()` lifecycle**

```python
# tests/unit/plugins/sinks/test_chroma_sink.py
"""Tests for ChromaSink plugin lifecycle and write operations."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.plugins.sinks.chroma_sink import ChromaSink


def _make_config() -> dict[str, Any]:
    return {
        "collection": "test-collection",
        "mode": "persistent",
        "persist_directory": "./test_chroma",
        "distance_function": "cosine",
        "field_mapping": {
            "document": "text",
            "id": "doc_id",
            "metadata": ["topic"],
        },
        "on_duplicate": "overwrite",
        "schema": {
            "mode": "fixed",
            "fields": ["doc_id: str", "text: str", "topic: str"],
        },
    }


class TestChromaSinkOnStart:
    def test_constructs_persistent_client(self) -> None:
        sink = ChromaSink(_make_config())
        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"
        mock_ctx.telemetry_emit = MagicMock()

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = MagicMock()

            sink.on_start(mock_ctx)

            mock_chromadb.PersistentClient.assert_called_once()

    def test_on_start_failure_raises(self) -> None:
        sink = ChromaSink(_make_config())
        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.side_effect = RuntimeError("Connection refused")

            with pytest.raises(RuntimeError, match="Connection refused"):
                sink.on_start(mock_ctx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: FAIL — `ChromaSink` not defined or incomplete

- [ ] **Step 3: Implement `ChromaSink` class with `on_start()`, `close()`**

Add to `chroma_sink.py` after the config classes:

```python
import hashlib
import time
from collections.abc import Mapping

import chromadb

from elspeth.contracts.contexts import LifecycleContext, SinkContext
from elspeth.contracts.enums import CallStatus, CallType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


class ChromaSink(BaseSink):
    """Write pipeline rows into a ChromaDB collection."""

    name = "chroma_sink"
    plugin_version = "1.0.0"
    supports_resume = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._config = ChromaSinkConfig.from_dict(config)
        self._schema_class = create_schema_from_config(
            self._config.schema_config,
            "ChromaSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._run_id: str | None = None
        self._telemetry_emit = None
        self._total_written = 0
        self._total_bytes = 0

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit

        if self._config.mode == "persistent":
            self._client = chromadb.PersistentClient(
                path=self._config.persist_directory,
            )
        else:
            self._client = chromadb.HttpClient(
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.ssl,
            )

        self._collection = self._client.get_or_create_collection(
            name=self._config.collection,
            metadata={"hnsw:space": self._config.distance_function},
        )

    def flush(self) -> None:
        pass

    def on_complete(self, ctx: LifecycleContext) -> None:
        if self._telemetry_emit is not None:
            self._telemetry_emit({
                "event": "chroma_sink_complete",
                "collection": self._config.collection,
                "total_written": self._total_written,
                "total_bytes": self._total_bytes,
            })

    def close(self) -> None:
        self._client = None
        self._collection = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "feat: add ChromaSink class with on_start/close lifecycle"
```

---

### Task 3: `ChromaSink.write()` — Overwrite Mode

**Files:**
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py`

- [ ] **Step 1: Write test for overwrite mode write**

```python
# Append to tests/unit/plugins/sinks/test_chroma_sink.py

class TestChromaSinkWriteOverwrite:
    def test_upserts_rows_with_correct_mapping(self) -> None:
        sink = ChromaSink(_make_config())
        mock_collection = MagicMock()
        sink._collection = mock_collection
        sink._run_id = "test-run"

        mock_ctx = MagicMock(spec_set=["run_id", "record_call", "operation_id"])
        mock_ctx.run_id = "test-run"
        mock_ctx.operation_id = "op-1"
        mock_ctx.record_call = MagicMock()

        rows = [
            {"doc_id": "d1", "text": "Hello world", "topic": "greeting"},
            {"doc_id": "d2", "text": "Goodbye", "topic": "farewell"},
        ]

        result = sink.write(rows, mock_ctx)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs.kwargs["ids"] == ["d1", "d2"]
        assert call_kwargs.kwargs["documents"] == ["Hello world", "Goodbye"]
        assert call_kwargs.kwargs["metadatas"] == [
            {"topic": "greeting"},
            {"topic": "farewell"},
        ]
        assert result is not None

    def test_records_audit_call(self) -> None:
        sink = ChromaSink(_make_config())
        mock_collection = MagicMock()
        sink._collection = mock_collection
        sink._run_id = "test-run"

        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"
        mock_ctx.record_call = MagicMock()

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]
        sink.write(rows, mock_ctx)

        mock_ctx.record_call.assert_called_once()
        call_kwargs = mock_ctx.record_call.call_args.kwargs
        assert call_kwargs["provider"] == "chromadb"

    def test_returns_artifact_with_content_hash(self) -> None:
        sink = ChromaSink(_make_config())
        mock_collection = MagicMock()
        sink._collection = mock_collection
        sink._run_id = "test-run"

        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]
        result = sink.write(rows, mock_ctx)

        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA-256 hex
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py::TestChromaSinkWriteOverwrite -v`
Expected: FAIL — `write()` not implemented

- [ ] **Step 3: Implement `write()` for overwrite mode**

Add to the `ChromaSink` class:

```python
    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> ArtifactDescriptor:
        chroma_url = SanitizedDatabaseUrl.from_raw_url(
            f"chromadb://{self._config.collection}"
        )

        if not rows:
            return ArtifactDescriptor.for_database(
                url=chroma_url,
                table=self._config.collection,
                content_hash=hashlib.sha256(b"").hexdigest(),
                payload_size=0,
                row_count=0,
            )

        fm = self._config.field_mapping
        ids = [row[fm.id] for row in rows]
        documents = [row[fm.document] for row in rows]
        metadatas = [
            {field: row[field] for field in fm.metadata} for row in rows
        ]

        payload = canonical_json({"ids": ids, "documents": documents, "metadatas": metadatas})
        payload_bytes = payload.encode("utf-8")
        content_hash = hashlib.sha256(payload_bytes).hexdigest()
        payload_size = len(payload_bytes)

        start_time = time.perf_counter()
        try:
            if self._config.on_duplicate == "overwrite":
                self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            elif self._config.on_duplicate == "skip":
                existing = self._collection.get(ids=ids)
                existing_ids = set(existing["ids"])
                new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
                if new_indices:
                    self._collection.add(
                        ids=[ids[i] for i in new_indices],
                        documents=[documents[i] for i in new_indices],
                        metadatas=[metadatas[i] for i in new_indices],
                    )
            elif self._config.on_duplicate == "error":
                existing = self._collection.get(ids=ids)
                existing_ids = set(existing["ids"])
                duplicates = [id_ for id_ in ids if id_ in existing_ids]
                if duplicates:
                    raise RuntimeError(
                        f"Duplicate document IDs in collection "
                        f"'{self._config.collection}': {duplicates}"
                    )
                self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

            latency_ms = (time.perf_counter() - start_time) * 1000
        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            ctx.record_call(
                call_type=CallType.VECTOR,
                status=CallStatus.ERROR,
                request_data={
                    "operation": self._config.on_duplicate.upper(),
                    "collection": self._config.collection,
                    "row_count": len(rows),
                },
                error={"type": "write_error"},
                latency_ms=latency_ms,
                provider="chromadb",
            )
            raise

        try:
            ctx.record_call(
                call_type=CallType.VECTOR,
                status=CallStatus.SUCCESS,
                request_data={
                    "operation": self._config.on_duplicate.upper(),
                    "collection": self._config.collection,
                    "row_count": len(rows),
                    "document_ids": ids,
                },
                response_data={"rows_written": len(rows)},
                latency_ms=latency_ms,
                provider="chromadb",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful ChromaDB write to audit trail "
                f"(collection={self._config.collection!r}, row_count={len(rows)}). "
                f"Write completed but audit record is missing."
            ) from exc

        self._total_written += len(rows)
        self._total_bytes += payload_size

        return ArtifactDescriptor.for_database(
            url=chroma_url,
            table=self._config.collection,
            content_hash=content_hash,
            payload_size=payload_size,
            row_count=len(rows),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "feat: implement ChromaSink.write() with overwrite/skip/error modes"
```

---

### Task 4: `ChromaSink.write()` — Skip and Error Mode Tests

**Files:**
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1: Write tests for skip mode**

```python
# Append to tests/unit/plugins/sinks/test_chroma_sink.py

class TestChromaSinkWriteSkip:
    def _make_skip_config(self) -> dict[str, Any]:
        config = _make_config()
        config["on_duplicate"] = "skip"
        return config

    def test_skip_filters_existing_ids(self) -> None:
        sink = ChromaSink(self._make_skip_config())
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink._collection = mock_collection
        sink._run_id = "test-run"


        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"

        rows = [
            {"doc_id": "d1", "text": "Existing", "topic": "t"},
            {"doc_id": "d2", "text": "New", "topic": "t"},
        ]
        sink.write(rows, mock_ctx)

        mock_collection.add.assert_called_once()
        add_kwargs = mock_collection.add.call_args.kwargs
        assert add_kwargs["ids"] == ["d2"]
        assert add_kwargs["documents"] == ["New"]


class TestChromaSinkWriteError:
    def _make_error_config(self) -> dict[str, Any]:
        config = _make_config()
        config["on_duplicate"] = "error"
        return config

    def test_error_mode_raises_on_duplicates(self) -> None:
        sink = ChromaSink(self._make_error_config())
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1", "d3"]}
        sink._collection = mock_collection
        sink._run_id = "test-run"


        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"

        rows = [
            {"doc_id": "d1", "text": "A", "topic": "t"},
            {"doc_id": "d2", "text": "B", "topic": "t"},
            {"doc_id": "d3", "text": "C", "topic": "t"},
        ]

        with pytest.raises(RuntimeError, match="d1.*d3|d3.*d1"):
            sink.write(rows, mock_ctx)

        mock_collection.add.assert_not_called()

    def test_error_mode_succeeds_when_no_duplicates(self) -> None:
        sink = ChromaSink(self._make_error_config())
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        sink._collection = mock_collection
        sink._run_id = "test-run"


        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"

        rows = [{"doc_id": "d1", "text": "A", "topic": "t"}]
        result = sink.write(rows, mock_ctx)

        mock_collection.add.assert_called_once()
        assert result is not None
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "test: add skip and error mode tests for ChromaSink.write()"
```

---

### Task 5: `AuditIntegrityError` Path and `flush()` No-Op

**Files:**
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1: Write test for `AuditIntegrityError`**

```python
# Append to tests/unit/plugins/sinks/test_chroma_sink.py

from elspeth.contracts.errors import AuditIntegrityError


class TestChromaSinkAuditIntegrity:
    def test_audit_recording_failure_raises_audit_integrity_error(self) -> None:
        sink = ChromaSink(_make_config())
        mock_collection = MagicMock()
        sink._collection = mock_collection
        sink._run_id = "test-run"


        mock_ctx = MagicMock()
        mock_ctx.run_id = "test-run"
        mock_ctx.record_call.side_effect = RuntimeError("DB write failed")

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]

        with pytest.raises(AuditIntegrityError, match="audit"):
            sink.write(rows, mock_ctx)

        # The ChromaDB write still happened
        mock_collection.upsert.assert_called_once()


class TestChromaSinkFlush:
    def test_flush_is_noop(self) -> None:
        sink = ChromaSink(_make_config())
        mock_collection = MagicMock()
        sink._collection = mock_collection

        sink.flush()

        # Verify no ChromaDB methods were called
        mock_collection.assert_not_called()
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "test: add AuditIntegrityError and flush no-op tests for ChromaSink"
```

---

### Task 6: Integration Test — CSV to ChromaSink Pipeline

**Files:**
- Create: `tests/integration/plugins/sinks/test_chroma_sink_pipeline.py`

- [ ] **Step 1: Write integration test with real ephemeral ChromaDB**

```python
# tests/integration/plugins/sinks/test_chroma_sink_pipeline.py
"""Integration test: CSV source → ChromaSink with real ChromaDB.

Uses persistent mode with tmp_path for deterministic CI behaviour.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import chromadb
import pytest

from elspeth.plugins.sinks.chroma_sink import ChromaSink


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "documents.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "text_content", "topic"])
        writer.writeheader()
        writer.writerow({"doc_id": "d1", "text_content": "Photosynthesis converts light to energy", "topic": "biology"})
        writer.writerow({"doc_id": "d2", "text_content": "Earthquakes release seismic waves", "topic": "geology"})
        writer.writerow({"doc_id": "d3", "text_content": "DNA encodes genetic information", "topic": "biology"})
    return csv_path


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma_data"


class TestChromaSinkIntegration:
    def test_documents_written_to_collection(
        self, sample_csv: Path, chroma_dir: Path
    ) -> None:
        """Verify documents are written with correct IDs, content, and metadata."""
        # This test constructs the sink directly with a real ChromaDB
        # to verify the write path without full orchestrator setup.
        # Full pipeline integration is in the end-to-end smoke test (sub-plan 5).

        config: dict[str, Any] = {
            "collection": "test-docs",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "distance_function": "cosine",
            "field_mapping": {
                "document": "text_content",
                "id": "doc_id",
                "metadata": ["topic"],
            },
            "on_duplicate": "overwrite",
            "schema": {
                "mode": "fixed",
                "fields": [
                    "doc_id: str",
                    "text_content: str",
                    "topic: str",
                ],
            },
        }

        sink = ChromaSink(config)

        # Simulate lifecycle
        from unittest.mock import MagicMock

        mock_ctx = MagicMock()
        mock_ctx.run_id = "integration-test-run"
        mock_ctx.telemetry_emit = MagicMock()

        sink.on_start(mock_ctx)

        rows = [
            {"doc_id": "d1", "text_content": "Photosynthesis converts light to energy", "topic": "biology"},
            {"doc_id": "d2", "text_content": "Earthquakes release seismic waves", "topic": "geology"},
            {"doc_id": "d3", "text_content": "DNA encodes genetic information", "topic": "biology"},
        ]

        write_ctx = MagicMock()
        write_ctx.run_id = "integration-test-run"
        write_ctx.record_call = MagicMock()

        artifact = sink.write(rows, write_ctx)

        # Verify artifact
        assert artifact.row_count == 3
        assert artifact.content_hash is not None
        assert len(artifact.content_hash) == 64

        # Verify audit call was recorded
        write_ctx.record_call.assert_called_once()
        call_kwargs = write_ctx.record_call.call_args.kwargs
        assert call_kwargs["provider"] == "chromadb"
        assert call_kwargs["request_data"]["row_count"] == 3

        sink.close()

        # Verify data in ChromaDB directly
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("test-docs")
        assert collection.count() == 3

        results = collection.get(ids=["d1", "d2", "d3"], include=["documents", "metadatas"])
        assert "Photosynthesis" in results["documents"][0]
        assert results["metadatas"][0]["topic"] == "biology"

    def test_upsert_is_idempotent(
        self, chroma_dir: Path
    ) -> None:
        """Verify overwrite mode can be re-run without errors."""
        config: dict[str, Any] = {
            "collection": "idempotent-test",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "field_mapping": {"document": "text", "id": "id", "metadata": []},
            "on_duplicate": "overwrite",
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        }

        from unittest.mock import MagicMock

        for _ in range(3):
            sink = ChromaSink(config)
            ctx = MagicMock()
            ctx.run_id = "idem-run"
            ctx.telemetry_emit = MagicMock()
            sink.on_start(ctx)

            write_ctx = MagicMock()
            write_ctx.run_id = "idem-run"
            write_ctx.record_call = MagicMock()
            sink.write([{"id": "d1", "text": "Hello"}], write_ctx)
            sink.close()

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("idempotent-test")
        assert collection.count() == 1
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/integration/plugins/sinks/test_chroma_sink_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/plugins/sinks/test_chroma_sink_pipeline.py
git commit -m "test: add ChromaSink integration tests with real ephemeral ChromaDB"
```

---

### Task 7: Type Checking, Linting, Full Test Suite

**Files:** None new — verification only.

- [ ] **Step 1: Run type checker on new files**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/sinks/chroma_sink.py`
Expected: PASS

- [ ] **Step 2: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/sinks/chroma_sink.py`
Expected: PASS

- [ ] **Step 3: Run full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -x -q`
Expected: PASS (no regressions)

- [ ] **Step 4: Run full integration test suite**

Run: `.venv/bin/python -m pytest tests/integration/ -x -q`
Expected: PASS (no regressions)

- [ ] **Step 5: Verify plugin auto-discovery**

Run: `elspeth plugins list | grep chroma`
Expected: `chroma_sink` appears in the sink plugins list

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address type/lint issues in ChromaSink"
```
